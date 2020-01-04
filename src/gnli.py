""" Natural langauge inference model wrapper for BERT and ROBERTA """
import logging
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
	linear_layer: ``FeedForward`` required.
		A linear layer that casts the output of BART to match the number of possible labels.
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
				 linear_layer: FeedForward,
				 discriminative_loss_weight: float = 0,
				 vocab: Vocabulary = Vocabulary(),
				 initializer: InitializerApplicator = InitializerApplicator()) -> None:
		super(GNLI, self).__init__(vocab)
		assert pretrained_model in ['bart.large']
		assert discriminative_loss_weight >= 0 and discriminative_loss_weight <= 1

		self._bart = torch.hub.load('pytorch/fairseq', pretrained_model).model
		self._linear_layer = linear_layer
		self._discriminative_loss_weight = discriminative_loss_weight

		self.metrics = {'accuracy': CategoricalAccuracy()}
		self.ce_loss = torch.nn.CrossEntropyLoss()
		initializer(self)

		# Log the number of trainable parameters in the model
		number_params = sum([numpy.prod(p.size()) for p in list(self.parameters()) if p.requires_grad])
		logger.info('Number of trainable model parameters: %d', number_params)

	@overrides
	def forward(self, 
				src: torch.Tensor, 					# src.size()	 			= [batch_size, premise_len]
				src_lengths: torch.Tensor, 			# src_lengths.size() 		= [batch_size]
				prev_output_tokens: torch.Tensor, 	# prev_output_tokens.size() = [batch_size, hypothesis_len]
				target: torch.Tensor,				# target.size() 			= [batch_size, hypothesis_len]	
				target_lengths: torch.Tensor, 		# target_lengths.size()		= [batch_size]
				label: torch.Tensor = None,	 		# label.size() 				= [batch_size]
				metadata = None):
		

		
		# logits.size() = [batch_size, 3]
		logits = self._linear_layer(cls_embed).float()
		
		output_dict = {'class_probabilities': torch.nn.functional.softmax(logits, dim=-1),
					   'predicted_label': torch.max(logits, dim=-1)[1],
					   'metadata': metadata}

		if label is not None:		
			label = label.long()
			output_dict['label'] = label

			# Discriminative Loss

			# Generative Loss

			# Mix the two losses
			self.metrics['accuracy'](logits, label)
			output_dict['loss'] = self.ce_loss(logits, label)

		return output_dict

	@overrides
	def get_metrics(self, reset: bool = False) -> Dict[str, float]:
		return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}