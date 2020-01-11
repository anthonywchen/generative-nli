""" Natural langauge inference model wrapper for BERT and ROBERTA """
import logging
import math
import numpy
from overrides import overrides
from pytorch_transformers import BertModel, RobertaModel
import torch
from torch.nn import CrossEntropyLoss
from typing import Dict

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.modules.feedforward import FeedForward
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator
from allennlp.training.metrics.average import Average
from allennlp.training.metrics.categorical_accuracy import CategoricalAccuracy

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

import random
@Model.register("gnli")
class GNLI(Model):
	"""	
	Parameters
	----------
	pretrained_model: ``str`` required.
		The name of the pretrained model used to intiailize the BART model.
	discriminative_loss_weight: ``float``
		This float specifies how strongly we weigh the discriminative loss 
		in the affine combination between the generative and the discriminative loss.
		The generative loss is the autoregressive loss with the correct label.
		The discriminative loss is the cross-entropy loss of p(h|p, label) over all
		labels.
		A value of 0 means we only rely on the generative loss. 
		A value of 1 means we only rely on the discriminative loss.
	vocab: ``Vocabulary`` 
		A vocabulary file that should be empty, since BART has its own vocabulary.
	"""
	@property
	def vocab_size(self):
		return self._bart.encoder.embed_tokens.num_embeddings

	def __init__(self, 
				 pretrained_model: str,
				 discriminative_loss_weight: float = 0,
				 vocab: Vocabulary = Vocabulary(),
				 initializer: InitializerApplicator = InitializerApplicator()) -> None:
		super(GNLI, self).__init__(vocab)
		# Check the arguments of `__init__()`.
		assert pretrained_model in ['bart.large']
		assert discriminative_loss_weight >= 0 and discriminative_loss_weight <= 1

		# Load in BART and extend the embeddings layer by three for the label embeddings.
		self._bart = torch.hub.load('pytorch/fairseq', pretrained_model).model
		self._extend_embeddings()

		# Ignore padding indices when calculating generative loss.
		self._generative_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self._bart.encoder.padding_idx)
		self._discriminative_loss_fn = torch.nn.NLLLoss()
		self._discriminative_loss_weight = discriminative_loss_weight
		self.metrics = {'accuracy': CategoricalAccuracy(), 
						'generative_loss': Average(), 
						'discriminative_loss': Average()}

		initializer(self)
		number_params = sum([numpy.prod(p.size()) for p in list(self.parameters()) if p.requires_grad])
		logger.info('Number of trainable model parameters: %d', number_params)

	def _extend_embeddings(self):
		""" Extends the embeddings in the encoder and decoder by three for the 
		class embeddings.

		Code based on HuggingFace Transformer repository.
		"""
		old_embeddings = self._bart.encoder.embed_tokens
		old_num_tokens, embedding_dim = old_embeddings.weight.size()

		# Build new embeddings and copy the word embeddings from the previous weights
		new_num_tokens = old_num_tokens + 3
		new_embeddings = torch.nn.Embedding(new_num_tokens, embedding_dim, padding_idx=old_embeddings.padding_idx)
		new_embeddings.to(old_embeddings.weight.device)
		new_embeddings.weight.data[:old_num_tokens, :] = old_embeddings.weight.data[:old_num_tokens, :]

		# Set the encoder to use the new embeddings and tie the encoder and decoder embeddings.
		self._bart.encoder.embed_tokens = new_embeddings
		self._bart.decoder.embed_tokens = self._bart.encoder.embed_tokens
		assert self._bart.decoder.embed_tokens == self._bart.encoder.embed_tokens

	@overrides
	def forward(self, 
				src: torch.Tensor, 					# src.size()	 			= [batch_size, 3, premise_length]
				src_lengths: torch.Tensor, 			# src_lengths.size() 		= [batch_size, 3]
				prev_output_tokens: torch.Tensor, 	# prev_output_tokens.size() = [batch_size, 3, hypothesis_length]
				target: torch.Tensor,				# target.size() 			= [batch_size, hypothesis_length]	
				target_lengths: torch.Tensor, 		# target_lengths.size()		= [batch_size]
				label: torch.Tensor = None,	 		# label.size() 				= [batch_size]
				metadata = None):
		batch_size, num_classes, premise_length = src.size()
		hypothesis_length = target.size(-1)
		assert num_classes == 3

		## Before feeding tensors through BART, merge the batch size and number of classes dimensions
		src_resized = src.resize(batch_size*num_classes, premise_length)
		src_lengths_resized = src_lengths.resize(batch_size*num_classes)
		prev_output_tokens_resized = prev_output_tokens.resize(batch_size*num_classes, hypothesis_length)
		
		## Feed tensors through BART model, which returns the logits from the decoder.
		# A set of logits is returned for each token in `prev_output_tokens` over the vocabulary.
		# decoder_logits.size() = [batch_size*3, hypothesis_length, vocab_size]
		decoder_logits, _ = self._bart(src_tokens=src_resized, src_lengths=src_lengths_resized, prev_output_tokens=prev_output_tokens_resized)
		decoder_logits = decoder_logits.resize(batch_size, num_classes, hypothesis_length, self.vocab_size)

		## Calculate the probability at each decoder timestep of seeing the next hypothesis token (i.e. the target token).
		# This is useful for visualizing what token most strongly influences the classification decision.
		
		# First calculate probabilities over the vocabulary at each decoder timestep.
		# Add a small constant to prevent tokens from have a probability of 0.
		# decoder_probabilties.size() = [batch_size, 3, hypothesis_length, vocab_size]
		decoder_probabilties = 1e-15 + torch.nn.functional.softmax(decoder_logits, dim=-1)

		# Expand size of target to match the # of dimensions of `decoder_probabilties`.
		# target_expanded.size() = [batch_size, 3, hypothesis_length, 1]
		target_expanded = target.unsqueeze(1).repeat(1, num_classes, 1).unsqueeze(-1)
		assert list(target_expanded.size()) == [batch_size, num_classes, hypothesis_length, 1]

		# Grab the probabilities of the target tokens.
		# target_decoder_probabilties.size() = [batch_size, 3, hypothesis_length]
		target_decoder_probabilties = torch.gather(decoder_probabilties, dim=-1, index=target_expanded).squeeze(-1)
		assert list(target_decoder_probabilties.size()) == [batch_size, num_classes, hypothesis_length]

		## Calculate the logits for the classes.
		# Add a small amount to each class logit since we directly calculate the probs from it and
		# if all logits are 0, we will get 0's in the loss function.
		# class_logits.size() = [batch_size, 3]
		class_logits = 1e-15 + self.calculate_class_logits(target_decoder_probabilties, target_lengths)
		class_logits_sum = torch.sum(class_logits, dim=-1)
		class_logits_sum = class_logits_sum.unsqueeze(-1).repeat(1, num_classes)

		# Calculate the class probabilities as the class logits divided by the sum of the other class logits
		class_probabilities = class_logits/class_logits_sum

		output_dict = {'decoder_probabilties': decoder_probabilties,
					   'target_decoder_probabilities': target_decoder_probabilties,
					   'class_logits': class_logits,
					   'class_probabilities': class_probabilities,
					   'predicted_label': torch.max(class_logits, dim=-1)[1],
					   'target': target,
					   'target_lengths': target_lengths,
					   'metadata': metadata}
		
		if random.random() < 0.01:
			print('\tprob', class_probabilities.tolist())
			print('\tlogit', class_logits.tolist())
			print('\tlabel', label.tolist())
			
		if label is not None:		
			label = label.long()
			target = target.long()

			output_dict['label'] = label
			self.metrics['accuracy'](class_logits, label)
			
			###### Discriminative Loss ######
			# Discriminative loss is the negative log likelihood over the class probabilities
			discriminative_loss = self._discriminative_loss_fn(torch.log(class_probabilities), label)
			output_dict['discriminative_loss'] = discriminative_loss.item()
			self.metrics['discriminative_loss'](discriminative_loss.item())

			###### Generative Loss ######
			# Extract from the decoder logits the logits over the correct class
			# Expand the labels to match the # of dimensions of the decoder logits
			# label_expanded.size() = [batch_size, 1, hypothesis_length, vocab_size]
			label_expanded = label.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, hypothesis_length, self.vocab_size)
			assert list(label_expanded.size()) == [batch_size, 1, hypothesis_length, self.vocab_size]
			# correct_class_decoder_logits.size() = [batch_size, hypothesis_length, vocab_size]
			correct_class_decoder_logits = torch.gather(decoder_logits, dim=1, index=label_expanded).squeeze(1)
			assert list(correct_class_decoder_logits.size()) == [batch_size, hypothesis_length, self.vocab_size]

			# Resize the correct class decoder logits and the targets to be 2D and 1D respectively
			# before calculating the generative loss since CrossEntropyLoss() expects these dimensions.
			correct_class_decoder_logits_resized = correct_class_decoder_logits.resize(batch_size*hypothesis_length, self.vocab_size)
			target_resized = target.resize(batch_size*hypothesis_length)

			generative_loss = self._generative_loss_fn(correct_class_decoder_logits_resized, target_resized)
			output_dict['generative_loss'] = generative_loss.item()
			self.metrics['generative_loss'](generative_loss.item())

			###### Mix the disciminative and generative losses via an affine combination ######
			if self._discriminative_loss_weight == 0:
				output_dict['loss'] = generative_loss
			elif self._discriminative_loss_weight == 1:
				output_dict['loss'] = discriminative_loss
			else:
				output_dict['loss'] = self._discriminative_loss_weight*discriminative_loss + (1-self._discriminative_loss_weight)*generative_loss

		return output_dict

	def old_calculate_class_logits(self, hypothesis_probabilities, target_lengths):
		""" Calculates the class logits from the probabilties of the hypothesis tokens.
		
		Essentially, this calculates p(hypothesis | premise, class) for the three classes
		by calculating the autoregressive probability over the hypothesis tokens. 
		p(hypothesis | premise, class ) is treated as the logit for that class.
		
		HOWEVER, we cannot calculate p(hypothesis | .) directly, since continually multiplying
		probabilties will quickly results in underflow. Even if it doesn't result in underflow, 
		doing a softmax on logits that are very small basically results in a uniform distributions.
		
		WE SOLVE THIS PROBLEM IN A HACKY WAY. 

		We rely on the fact that when calculating the softmax, multiplying
		the inputs to the softmax by a constant will preserve the resulting probabilties. 

		i.e. e^x_i / (e^x_1 + e^x_2 ... ) = e^(c*x_i) / (e^(c*x_1) + e^(c*x_2) ... )

		We set this constant as ten to the negative base10 value of the largest 
		probabiltiy would be for the classes if we did directly compute p(h | .) 
		(`10**min_base10_factor`). 

		i.e. largest_prob = argmax class' p(hypothesis | premise, class')
			 min_base10_factor = -1 * ceil(log_base_10(largest_prob))
			 constant = 10**min_base10_factor
		
		HOWEVER, we cannot use this constant directly, since it itself may result in an overflow if 
		`min_base10_factor` is very large.
		
		Thus, we iterate through each token and multiply its probability 
		by 10 to the negative base10 of its probability while that amount we multiply by has not exceeded 
		`10**min_base10_factor`. This ensures that at the end we have multiplied by our constant while
		preventing overflow and keeping the resulting softmax distribution sharp.
		
		The resulting logits preseve the probabilites when passed through a softmax function.
		"""
		batch_size, num_classes, _ = hypothesis_probabilities.size()

		# Initialize logits as all 1's. We will multiply by the target token probs.
		class_logits = torch.ones(batch_size, num_classes)
		class_logits = class_logits.type_as(hypothesis_probabilities)
		
		# Each data point has a different # of hypothesis tokens so handle so data point iteratively
		for batch_entry in range(batch_size):
			target_len = target_lengths[batch_entry].item()
			
			# This is the smallest (negative) base 10 of p(h| premise, label) for this batch entry across all labels.
			base10_factors = torch.sum(torch.floor(-1*torch.log10(hypothesis_probabilities[batch_entry, :, :target_len])), dim=-1)
			min_base10_factor = torch.min(base10_factors).item()

			for label_entry in range(num_classes):
				multiplicative_factor = min_base10_factor
				
				for hypothesis_entry in range(target_len):
					cur_token_prob = hypothesis_probabilities[batch_entry, label_entry, hypothesis_entry]
					
					cur_base10_factor = torch.floor(-1*torch.log10(cur_token_prob)).item()
					cur_base10_factor = min(cur_base10_factor, multiplicative_factor)

					# Multiply current hypothesis token probability by its negative base10 value to prevent underflow
					class_logits[batch_entry, label_entry] *= cur_token_prob*(10**cur_base10_factor)

					multiplicative_factor -= cur_base10_factor
				
				# Check that we have multiplied the label logit by `10**min_base10_factor`
				# print(multiplicative_factor)
				assert multiplicative_factor == 0
		return class_logits

	def calculate_class_logits(self, hypothesis_probabilities, target_lengths):
		""" Calculates the class logits from the probabilties of the hypothesis tokens.
		
		Essentially, this calculates p(hypothesis | premise, class) for the three classes
		by calculating the autoregressive probability over the hypothesis tokens. 
		p(hypothesis | premise, class ) is treated as the logit for that class.
		"""
		batch_size, num_classes, _ = hypothesis_probabilities.size()
		log_hypothesis_probabilities = torch.log(hypothesis_probabilities)

		# Sum the probabiltiies of the hypothesis tokens up to the length of the hypothesis (without padding)
		log_class_logits = torch.zeros(batch_size, num_classes)
		log_class_logits = log_class_logits.type_as(hypothesis_probabilities)

		for batch_entry in range(batch_size):
			cur_target_len = target_lengths[batch_entry].item()
			log_class_logits[batch_entry] = torch.sum(log_hypothesis_probabilities[batch_entry, :, :cur_target_len], dim=-1)
			
		# scaling.size() = [batch_size]
		scaling = torch.max(log_class_logits, dim=-1)[0]
		scaling = scaling.unsqueeze(-1).repeat(1, num_classes)
		log_class_logits -= scaling

		class_logits = torch.exp(log_class_logits)
		return class_logits

	@overrides
	def get_metrics(self, reset: bool = False) -> Dict[str, float]:
		return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}