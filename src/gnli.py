import logging
import math
import numpy
from overrides import overrides
import torch
from torch import nn
from typing import Dict

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
	disc_loss_weight: ``float``
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
	def label_size(self):
		return 3

	@property
	def vocab_size(self):
		return self.bart.encoder.embed_tokens.num_embeddings

	@property
	def hidden_dim(self):
		return self.bart.encoder.embed_tokens.embedding_dim

	@overrides
	def get_metrics(self, reset: bool = False) -> Dict[str, float]:
		return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}

	def __init__(self, 
		disc_loss_weight: float = 0,
		dropout: float = 0,
		vocab: Vocabulary = Vocabulary(),
		initializer: InitializerApplicator = InitializerApplicator()) -> None:

		super(GNLI, self).__init__(vocab)
		self.bart 			= torch.hub.load('pytorch/fairseq', 'bart.large').model
		self.dropout 		= nn.Dropout(p=dropout)
		self.label_vectors 	= torch.rand(1, self.label_size, 1, self.hidden_dim, requires_grad=True)
		self.projection 	= nn.Linear(self.hidden_dim*2, self.hidden_dim)

		assert 0 <= disc_loss_weight <= 1
		self.disc_loss_weight 	= disc_loss_weight
		self.disc_loss_fn 		= nn.NLLLoss()
		self.gen_loss_fn 		= nn.CrossEntropyLoss(ignore_index=self.bart.encoder.padding_idx)
		
		self.metrics = {'disc_loss': Average(), 
						'gen_loss': Average(),
						'accuracy': CategoricalAccuracy()}

		initializer(self)
		number_params = sum([numpy.prod(p.size()) for p in list(self.parameters()) if p.requires_grad])
		logger.info('Number of trainable model parameters: %d', number_params)

	@overrides
	def forward(self,
		src: torch.Tensor,                  # [batch_size, premise_length]
		src_lengths: torch.Tensor,          # [batch_size]
		prev_output_tokens: torch.Tensor,   # [batch_size, hypothesis_length]
		target: torch.Tensor,               # [batch_size, hypothesis_length]   
		target_lengths: torch.Tensor,       # [batch_size]
		label: torch.Tensor = None,         # [batch_size]
		metadata = None):

		batch_size, hypothesis_length = target.size()

		## Calculate the logits and probability over the vocabulary at each decoder time
		# decoder_logits.size() = [batch_size, 3, hypothesis_length, vocab_size]
		decoder_logits = self.bart_forward(src, src_lengths, prev_output_tokens)
		decoder_probabilties = 1e-15 + nn.functional.softmax(decoder_logits, dim=-1)

		## Calculate the probability of seeing the target tokens at each decoder time
		# target_expanded.size() = [batch_size, 3, hypothesis_length, 1]
		target_expanded = target.resize(batch_size, 1, hypothesis_length, 1).repeat(1, self.label_size, 1, 1)
		# target_decoder_probabilties.size() = [batch_size, 3, hypothesis_length]
		target_decoder_probabilities = decoder_probabilties.gather(dim=-1, index=target_expanded).squeeze(-1)

		class_probabilities =  self.calculate_class_probabilities(target_decoder_probabilities, target_lengths)

		output_dict = {'class_probabilities': class_probabilities,
					   'predicted_label': torch.max(class_probabilities, dim=-1)[1],
					   'target': target,
					   'target_lengths': target_lengths,
					   'target_decoder_probabilities': target_decoder_probabilities,
					   'metadata': metadata}
			
		if label is not None:
			label = label.long()
			target = target.long()

			output_dict['label'] = label
			self.metrics['accuracy'](class_probabilities, label)
			
			###### Discriminative Loss ######
			# Discriminative loss is the negative log likelihood over the class probabilities
			disc_loss = self.disc_loss_fn(torch.log(class_probabilities), label)
			output_dict['disc_loss'] = disc_loss.item()
			self.metrics['disc_loss'](disc_loss.item())

			###### Generative Loss ######
			## Extract from the decoder logits the logits of the target tokens over the correct label
			# Expand the labels to match the # of dimensions of the decoder logits
			# label_expanded.size() = [batch_size, 1, hypothesis_length, vocab_size]
			label_expanded = label.resize(batch_size, 1, 1, 1).repeat(1, 1, hypothesis_length, self.vocab_size)
			
			# correct_class_decoder_logits.size() = [batch_size, hypothesis_length, vocab_size]
			correct_class_decoder_logits = decoder_logits.gather(dim=1, index=label_expanded).squeeze(1)

			# Resize the correct class decoder logits and the targets to be 2D and 1D respectively
			# before calculating the generative loss since CrossEntropyLoss() expects these dimensions.
			correct_class_decoder_logits = correct_class_decoder_logits.resize(batch_size*hypothesis_length, self.vocab_size)
			target = target.resize(batch_size*hypothesis_length)

			gen_loss = self.gen_loss_fn(correct_class_decoder_logits, target)
			output_dict['gen_loss'] = gen_loss.item()
			self.metrics['gen_loss'](gen_loss.item())

			###### Mix the disciminative and generative losses via an affine combination ######
			if self.disc_loss_weight == 0:
				output_dict['loss'] = gen_loss
			elif self.disc_loss_weight == 1:
				output_dict['loss'] = disc_loss
			else:
				output_dict['loss'] = self.disc_loss_weight*disc_loss + (1-self.disc_loss_weight)*gen_loss

		return output_dict

	def bart_forward(self, src, src_lengths, prev_output_tokens):
		batch_size, hypothesis_length = prev_output_tokens.size()

		# Get the features over the decoder
		# features.size() = [batch_size, 3, hypothesis_length, hidden_dim]
		features = self.bart(src_tokens=src,
							 src_lengths=src_lengths,
							 prev_output_tokens=prev_output_tokens,
							 features_only=True)[0].unsqueeze(1).repeat_interleave(self.label_size, dim=1)

		# Project features with label vectors
		expanded_label_vectors 	= self.label_vectors.repeat(batch_size, 1, hypothesis_length, 1)
		proj_input 				= torch.cat((features, expanded_label_vectors), dim=-1)
		features 				= self.projection(self.dropout(proj_input))

		# Compute logits over the vocabulary and cast as float
		# logits.size() = [batch_size, label_size, hypothesis_length, vocab_size]
		logits = self.bart.decoder.output_layer(self.dropout(features))

		return logits.float()

	def calculate_class_probabilities(self, hypothesis_probabilities, target_lengths):
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
			
		log_unnormalized_class_probs = torch.sum(log_hypothesis_probabilities*hypothesis_mask, dim=-1)

		# scaling.size() = [batch_size]
		scaling = torch.max(log_unnormalized_class_probs, dim=-1)[0]
		scaling = scaling.unsqueeze(-1).repeat(1, self.label_size)
		log_unnormalized_class_probs = log_unnormalized_class_probs - scaling
		
		# unnormalized_class_probs.size() = [batch_size, 3]
		unnormalized_class_probs = 10**log_unnormalized_class_probs + 1e-15 

		# Calculate the (normalized) class probabilities by normalizing 
		normalization_value = torch.sum(unnormalized_class_probs, dim=-1)
		normalization_value = normalization_value.unsqueeze(-1).repeat(1, self.label_size)
		class_probabilities = unnormalized_class_probs/normalization_value
		assert math.isclose(torch.sum(class_probabilities).item(), batch_size, abs_tol=1e-5)

		return class_probabilities