import logging
import math
import numpy
from overrides import overrides
import torch
from typing import Dict

from allennlp.common import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator
from allennlp.training.metrics.average import Average
from allennlp.training.metrics.categorical_accuracy import CategoricalAccuracy

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


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

	@property
	def label_size(self):
		return 3

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
		assert self._bart.encoder.padding_idx == 1
		self._generative_loss_fn 		 = torch.nn.CrossEntropyLoss(ignore_index=self._bart.encoder.padding_idx)
		self._discriminative_loss_fn 	 = torch.nn.NLLLoss()
		self._discriminative_loss_weight = discriminative_loss_weight
		self.metrics 					 = {'accuracy': CategoricalAccuracy(), 'disc_loss': Average(), 'gen_loss': Average()}

		initializer(self)
		number_params = sum([numpy.prod(p.size()) for p in list(self.parameters()) if p.requires_grad])
		logger.info('Number of trainable model parameters: %d', number_params)

	@overrides
	def forward(self,
				src: torch.Tensor, 					# [batch_size, 3, premise_length]
				src_lengths: torch.Tensor, 			# [batch_size, 3]
				prev_output_tokens: torch.Tensor, 	# [batch_size, 3, hypothesis_length]
				target: torch.Tensor,				# [batch_size, hypothesis_length]	
				target_lengths: torch.Tensor, 		# [batch_size]
				label: torch.Tensor = None,	 		# [batch_size]
				metadata = None):
		batch_size, hypothesis_length = target.size()
		premise_length = src.size(-1)
		assert src.size(1) == self.label_size

		## Before feeding tensors through BART, merge the batch size and number of classes dimensions
		src 				= src.resize(batch_size*self.label_size, premise_length)
		src_lengths 		= src_lengths.resize(batch_size*self.label_size)
		prev_output_tokens 	= prev_output_tokens.resize(batch_size*self.label_size, hypothesis_length)
		
		## Feed tensors through BART model, which returns the logits from the decoder.
		# A set of logits is returned for each token in `prev_output_tokens` over the vocabulary.
		# Then unsqueeze the first dimension and convert to a float (for half-precision training)
		# decoder_logits.size() = [batch_size*3, hypothesis_length, vocab_size]
		decoder_logits, _ = self._bart(src_tokens=src,
									   src_lengths=src_lengths,
									   prev_output_tokens=prev_output_tokens)
		decoder_logits = decoder_logits.resize(batch_size, self.label_size, hypothesis_length, self.vocab_size).float()
		
		## Calculate the probability at each decoder timestep of seeing the next hypothesis token (i.e. the target token).
		# This is useful for visualizing what token most strongly influences the classification decision.
		
		# First calculate probabilities over the vocabulary at each decoder timestep.
		# Add a small constant to prevent tokens from have a probability of 0.
		# decoder_probabilties.size() = [batch_size, 3, hypothesis_length, vocab_size]
		decoder_probabilties = 1e-15 + torch.nn.functional.softmax(decoder_logits, dim=-1)

		# Expand size of `target` to match the # of dimensions of `decoder_probabilties`.
		# target_expanded.size() = [batch_size, 3, hypothesis_length, 1]
		target_expanded = target.unsqueeze(1).repeat(1, self.label_size, 1).unsqueeze(-1)

		# Grab the probabilities of the target tokens.
		# target_decoder_probabilties.size() = [batch_size, 3, hypothesis_length]
		target_decoder_probabilities = torch.gather(decoder_probabilties, dim=-1, index=target_expanded).squeeze(-1)
		assert target_decoder_probabilities.size() == (batch_size, self.label_size, hypothesis_length)

		## Calculate the logits for the classes.
		# Add a small constant to each class logit since we directly calculate the probs from it and
		# if all logits are 0, we will get 0's in the loss function.
		# class_logits.size() = [batch_size, 3]
		class_logits 	 = 1e-15 + self.calculate_class_logits(target_decoder_probabilities, target_lengths)
		class_logits_sum = torch.sum(class_logits, dim=-1)
		class_logits_sum = class_logits_sum.unsqueeze(-1).repeat(1, self.label_size)

		# Calculate the class probabilities by normalizing the class logit values
		class_probabilities = class_logits/class_logits_sum
		assert math.isclose(torch.sum(class_probabilities).item(), batch_size, abs_tol=1e-5)

		output_dict = {'class_logits': class_logits,
					   'class_probabilities': class_probabilities,
					   'predicted_label': torch.max(class_logits, dim=-1)[1],
					   'metadata': metadata,
					   'target_decoder_probabilities': target_decoder_probabilities,
					   'target': target,
					   'target_lengths': target_lengths}
			
		if label is not None:		
			label = label.long()
			target = target.long()

			output_dict['label'] = label
			self.metrics['accuracy'](class_logits, label)
			
			###### Discriminative Loss ######
			# Discriminative loss is the negative log likelihood over the class probabilities
			discriminative_loss = self._discriminative_loss_fn(torch.log(class_probabilities), label)
			output_dict['disc_loss'] = discriminative_loss.item()
			self.metrics['disc_loss'](discriminative_loss.item())

			###### Generative Loss ######
			## Extract from the decoder logits the logits over the correct class
			# Expand the labels to match the # of dimensions of the decoder logits
			# label_expanded.size() = [batch_size, 1, hypothesis_length, vocab_size]
			label_expanded = label.resize(batch_size, 1, 1, 1).repeat(1, 1, hypothesis_length, self.vocab_size)
			# correct_class_decoder_logits.size() = [batch_size, hypothesis_length, vocab_size]
			correct_class_decoder_logits = torch.gather(decoder_logits, dim=1, index=label_expanded).squeeze(1)
			assert list(correct_class_decoder_logits.size()) == [batch_size, hypothesis_length, self.vocab_size]

			# Resize the correct class decoder logits and the targets to be 2D and 1D respectively
			# before calculating the generative loss since CrossEntropyLoss() expects these dimensions.
			correct_class_decoder_logits = correct_class_decoder_logits.resize(batch_size*hypothesis_length, self.vocab_size)
			target = target.resize(batch_size*hypothesis_length)

			generative_loss = self._generative_loss_fn(correct_class_decoder_logits, target)
			output_dict['gen_loss'] = generative_loss.item()
			self.metrics['gen_loss'](generative_loss.item())

			###### Mix the disciminative and generative losses via an affine combination ######
			if self._discriminative_loss_weight == 0:
				output_dict['loss'] = generative_loss
			elif self._discriminative_loss_weight == 1:
				output_dict['loss'] = discriminative_loss
			else:
				output_dict['loss'] = self._discriminative_loss_weight*discriminative_loss + \
									  (1-self._discriminative_loss_weight)*generative_loss

		return output_dict

	def calculate_class_logits(self, hypothesis_probabilities, target_lengths):
		""" Calculates the class logits from the probabilties of the hypothesis tokens.
		
		Essentially, this calculates p(hypothesis | premise, class) for the three classes
		by calculating the autoregressive probability over the hypothesis tokens. 
		p(hypothesis | premise, class ) is treated as the logit for that class.
		"""
		batch_size = hypothesis_probabilities.size(0)
		log_hypothesis_probabilities = torch.log10(hypothesis_probabilities)

		# Sum the probabiltiies of the hypothesis tokens up to the length of the hypothesis (without padding)
		hypothesis_mask = torch.zeros(hypothesis_probabilities.size()).type_as(hypothesis_probabilities)
		for batch_entry, cur_target_len in enumerate(target_lengths):
			hypothesis_mask[batch_entry, :, :cur_target_len.item()] = 1
			
		log_class_logits = torch.sum(log_hypothesis_probabilities*hypothesis_mask, dim=-1)

		# scaling.size() = [batch_size]
		scaling = torch.max(log_class_logits, dim=-1)[0]
		scaling = scaling.unsqueeze(-1).repeat(1, self.label_size)
		log_class_logits = log_class_logits - scaling

		class_logits = 10**log_class_logits
		return class_logits

	@overrides
	def get_metrics(self, reset: bool = False) -> Dict[str, float]:
		return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}