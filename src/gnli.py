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
	def __init__(self, 
				 pretrained_model: str,
				 discriminative_loss_weight: float = 0,
				 vocab: Vocabulary = Vocabulary(),
				 initializer: InitializerApplicator = InitializerApplicator()) -> None:
		super(GNLI, self).__init__(vocab)
		assert pretrained_model in ['bart.large']
		assert discriminative_loss_weight >= 0 and discriminative_loss_weight <= 1

		# Load in BART and extend the embeddings layer by three for the label embeddings
		self._bart = torch.hub.load('pytorch/fairseq', pretrained_model).model
		self._extend_embeddings()

		self.discriminative_loss_weight = discriminative_loss_weight
		self.metrics = {'accuracy': CategoricalAccuracy()}
		self.ce_loss = torch.nn.CrossEntropyLoss()
		initializer(self)

		# Log the number of trainable parameters in the model
		number_params = sum([numpy.prod(p.size()) for p in list(self.parameters()) if p.requires_grad])
		logger.info('Number of trainable model parameters: %d', number_params)

	def _extend_embeddings(self):
		old_embeddings = self._bart.encoder.embed_tokens
		old_num_tokens, embedding_dim = old_embeddings.weight.size()

		# Build new embeddings and copy word embeddings from the previous weights
		new_num_tokens = old_num_tokens + 3
		new_embeddings = torch.nn.Embedding(new_num_tokens, embedding_dim)
		new_embeddings.to(old_embeddings.weight.device)
		new_embeddings.weight.data[:old_num_tokens, :] = old_embeddings.weight.data[:old_num_tokens, :]

		# Set the embeddings in encoder to new embeddings and tie the decoder embeddings
		self._bart.encoder.embed_tokens = new_embeddings
		self._bart.decoder.embed_tokens = self._bart.encoder.embed_tokens
		assert self._bart.decoder.embed_tokens == self._bart.encoder.embed_tokens

	@overrides
	def forward(self, 
				src: torch.Tensor, 					# src.size()	 			= [batch_size, 3, premise_length]
				src_lengths: torch.Tensor, 			# src_lengths.size() 		= [batch_size, 3]
				prev_output_tokens: torch.Tensor, 	# prev_output_tokens.size() = [batch_size, 3, hypothesis_length]
				target: torch.Tensor,				# target.size() 			= [batch_size, 3, hypothesis_length]	
				target_lengths: torch.Tensor, 		# target_lengths.size()		= [batch_size, 3]
				label: torch.Tensor = None,	 		# label.size() 				= [batch_size]
				metadata = None):
		premise_length = src.size(-1)
		batch_size, num_classes, hypothesis_length = target.size()
		assert num_classes == 3

		## Resize the input tensors so that the first and second dimension are merged
		src = src.resize(batch_size*num_classes, premise_length)
		src_lengths = src_lengths.resize(batch_size*num_classes)
		prev_output_tokens = prev_output_tokens.resize(batch_size*num_classes, hypothesis_length)
		
		## Feed through BART model, which returns the logits over `prev_output_tokens`
		#  target_token_logits.size() = [batch_size*3, hypothesis_length, vocab_size]
		all_token_logits, _ = self._bart(src_tokens=src, src_lengths=src_lengths, prev_output_tokens=prev_output_tokens)
		all_token_logits = all_token_logits.resize(batch_size, num_classes, hypothesis_length, self._bart.encoder.embed_tokens.num_embeddings)
		
		## Calculate hypothesis token probabilities.
		#  This is useful for visualizing what token most strongly influences the classification decision.
		all_token_probabilties = torch.nn.functional.softmax(all_token_logits, dim=-1)
		# hypothesis_probabilities.size() = [batch_size, num_classes, hypothesis_length]
		# Add a small constant to prevent 0 probabilties since that would set the prob of the hypothesis to 0.
		hypothesis_probabilities = 1e-15 + torch.gather(all_token_probabilties, dim=-1, index=target.unsqueeze(-1)).squeeze(-1)

		## Calculate the logits for the classes
		#  class_logits.size() = [batch_size, 3]
		class_logits = self.calculate_class_logits(hypothesis_probabilities, target_lengths)

		output_dict = {'hypothesis_probabilities': hypothesis_probabilities,
					   'class_probabilities': torch.nn.functional.softmax(class_logits, dim=-1),
					   'predicted_label': torch.max(class_logits, dim=-1)[1],
					   'target_lengths': target_lengths,
					   'metadata': metadata}

		if label is not None:		
			label = label.long()
			output_dict['label'] = label
			self.metrics['accuracy'](class_logits, label)
			
			# Discriminative Loss
			discriminative_loss = self.ce_loss(class_logits, label)

			# Generative Loss.
			# Iterate through batch entries, respecting the original hypothesis lengths (i.e. ignore padding).
			generative_loss = 0

			# Mix the disciminative and generative losses via an affine combination
			output_dict['loss'] = self.discriminative_loss_weight*discriminative_loss + (1-self.discriminative_loss_weight)*generative_loss

		return output_dict

	def calculate_class_logits(self, hypothesis_probabilities, target_lengths):
		""" Calculates the class logits from the probabilties of the hypothesis tokens.
		
		Essentially, this calculates p(hypothesis | premise, label) for the three classes
		by calculating the autoregressive probability over the hypothesis tokens as the logits for classes.
		
		HOWEVER, we cannot calculate p(hypothesis | .) directly, since continually multiplying
		probabilties will quickly results in underflow.
		
		WE SOLVE THIS PROBLEM IN A HACKY WAY. 

		We rely on the fact that when calculating the softmax, multiplying
		the inputs to the softmax by a constant will preserve the resulting probabilties. 

		We set this constant as ten to the negative base10 value of the largest 
		probabiltiy would be for the classes if we did directly compute p(h | .) 
		(`10**min_base10_factor`)
		
		HOWEVER, we cannot use this constant directly, since it itself may result in an overflow if 
		`min_base10_factor` is very large.
		
		Thus, we iterate through each token and multiply its probability 
		by 10 to the negative base10 of the probability while that amount we multiply by has not exceeded 
		`10**min_base10_factor`. This ensures that at the end we have multiplied by our constant while
		preventing overflow.
		
		The resulting logits preseve the probabilites when passed through a softmax function.
		"""
		batch_size, num_classes, _ = hypothesis_probabilities.size()
		
		# Initialize logits as all 1's. We will multiply by the target token probs.
		class_logits = torch.ones(batch_size, num_classes)
		
		# Each data point has a different # of hypothesis tokens so handle so data point iteratively
		for batch_entry in range(batch_size):
			target_len = target_lengths[batch_entry]
			
			# This is the smallest (negative) base 10 of p(h| premise, label) for this batch entry across all labels.
			min_base10_factor = torch.min(torch.sum(torch.ceil(-1*torch.log10(hypothesis_probabilities[batch_entry, :, :target_len])), dim=-1)).item()
			
			for label_entry in range(num_classes):
				multiplicative_factor = min_base10_factor
				
				for hypothesis_entry in range(target_len):
					cur_token_prob = hypothesis_probabilities[batch_entry, label_entry, hypothesis_entry]
					
					cur_base10_factor = math.ceil(math.log10(cur_token_prob.item())*-1)
					cur_base10_factor = min(cur_base10_factor, multiplicative_factor)
					
					# Multiply current hypothesis token probability by its negative base10 value to prevent underflow
					class_logits[batch_entry, label_entry] *= cur_token_prob*(10**cur_base10_factor)

					multiplicative_factor -= cur_base10_factor
				
				# Check that we have multiplied the label logit by `10**min_base10_factor`
				assert multiplicative_factor == 0
		return class_logits

	@overrides
	def get_metrics(self, reset: bool = False) -> Dict[str, float]:
		return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}