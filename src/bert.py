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
from allennlp.training.metrics.categorical_accuracy import  CategoricalAccuracy

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@Model.register("bertnli")
class BertNLI(Model):
	"""	
	Parameters
	----------
	pretrained_model: ``str`` required.
		The pretrained model used to intiailize the BERT model.
	linear_layer: ``FeedForward`` required.
		A linear layer that casts the output of BERT to match the number of possible labels.
	vocab: ``Vocabulary`` 
		A vocabulary file that should be empty, since BERT has its own vocabulary.
	"""
	def __init__(self, 
				 pretrained_model: str, 
				 linear_layer: FeedForward,
				 vocab: Vocabulary = Vocabulary(), 
				 initializer: InitializerApplicator = InitializerApplicator()) -> None:
		super(BertNLI, self).__init__(vocab)
		self._linear_layer = linear_layer
		self.metrics = {'accuracy': CategoricalAccuracy()}
		self.ce_loss = torch.nn.CrossEntropyLoss()
		
		self.model_class = pretrained_model.split('-')[0].lower()
		if self.model_class == 'bert':		
			self._bert_model = BertModel.from_pretrained(pretrained_model)
		elif self.model_class == 'roberta':	
			self._bert_model = RobertaModel.from_pretrained(pretrained_model)
		else:								
			raise ValueError('pretrained_model must either be roberta or bert')

		initializer(self)

		# Log the number of trainable parameters in the model
		number_params = sum([numpy.prod(p.size()) for p in list(self.parameters()) if p.requires_grad])
		logger.info('Number of trainable model parameters: %d', number_params)

	@overrides
	def forward(self, 
				input_ids: torch.Tensor, 		# input_ids.size() 		= [batch_size, seq_len]
				token_type_ids: torch.Tensor, 	# token_type_ids.size() = [batch_size, seq_len]
				attention_mask: torch.Tensor,	# attention_mask.size() = [batch_size, seq_len]
				label: torch.Tensor = None,	 	# label.size() 			= [batch_size]
				metadata = None):

		if self.model_class == 'bert':
			outputs = self._bert_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
		elif self.model_class == 'roberta':
			outputs = self._bert_model(input_ids=input_ids, attention_mask=attention_mask)

		# last_hidden_states.size() = [batch_size, seq_len, hidden_size]
		last_hidden_states = outputs[0]
		cls_embed = last_hidden_states[:,0,:]

		# logits.size() = [batch_size, 3]
		logits = self._linear_layer(cls_embed).float()
		class_probabiltiies = torch.nn.functional.softmax(logits, dim=-1)
		
		output_dict = {'class_probabilities': class_probabiltiies,
					   'predicted_label': torch.max(logits, dim=-1)[1],
					   'metadata': metadata}

		if label is not None:		
			label = label.long()
			self.metrics['accuracy'](logits, label) 
			output_dict['loss'] = self.ce_loss(class_probabiltiies, label)
			output_dict['label'] = label

		return output_dict

	@overrides
	def get_metrics(self, reset: bool = False) -> Dict[str, float]:
		return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}