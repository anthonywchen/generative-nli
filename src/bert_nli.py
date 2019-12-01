import logging
import numpy
from overrides import overrides
from pytorch_transformers import BertForPreTraining
import torch
from torch.nn import CrossEntropyLoss
from typing import Dict

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules.attention.linear_attention import LinearAttention
from allennlp.nn import InitializerApplicator
from allennlp.training.metrics.categorical_accuracy import  CategoricalAccuracy

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@Model.register("bertnli")
class BertNLI(Model):
	"""	
	Parameters
	----------
	bert_model: ``str`` required.
		The pretrained model used to intiailize the BERT model
	bert_config: ``str`` required.
	linear_attention: ``LinearAttention`` required.
		Used for `is_same_answer`. Linear layer combining the representations from 
		candidate and reference answer representations.
		While this is a `LinearAttention` function, we use it as a linear layer
		because of the flexibility it gives in terms of the combinations. 
		Therefore, we should not normalize the outputs.
	vocab: ``Vocabulary`` 
		A vocabulary file that should be empty, since BERT has its own vocabulary
	top_layer_only: ``bool``
		If false, we use a scalar mix as the output of the BERT model
	"""
	def __init__(self, 
				 bert_pretrained_model: str, 
				 classification_layer: LinearLayer,
				 vocab: Vocabulary = Vocabulary(), 
				 normalize_before_linear: bool = False, 
				 initializer: InitializerApplicator = InitializerApplicator()) -> None:
		super(BertNLI, self).__init__(vocab)

		initializer(self)

		# Log the number of trainable parameters in the model
		number_params = sum([numpy.prod(p.size()) for p in list(self.parameters()) if p.requires_grad])
		logger.info('Number of trainable model parameters: %d', number_params)

	@overrides
	def forward(self, 
				premise_ids: torch.Tensor, 		# premise_ids.size() 		= [batch_size, premise_seq_len]
				hypothesis_ids: torch.Tensor,  	# hypothesis_ids.size()   	= [batch_size, hypothesis_seq_len]
				label: torch.Tensor = None,	 	# label.size() 				= [batch_size]
				metadata = None):
		bs = premise_ids.size(0)

	@overrides
	def get_metrics(self, reset: bool = False) -> Dict[str, float]:
		return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}