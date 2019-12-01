from jsonlines import Reader
import logging
import numpy as np
from overrides import overrides
from pytorch_transformers import BertTokenizer, RobertaTokenizer
from typing import Callable, Dict, Iterable, Iterator, List

from allennlp.common import Params, Tqdm
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import ArrayField
from allennlp.data.fields.metadata_field import MetadataField
from allennlp.data.instance import Instance

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DatasetReader.register("nli")
class NLIDatasetReader(DatasetReader):
	def __init__(self,
				 batch_size,
				 num_epochs,
				 tokenizer_model,
				 tokenizer_name,
				 lazy=False) -> None:
		super().__init__(lazy)
		self._batch_size = batch_size
		self._num_epochs = num_epochs

		if tokenizer_model.lower() == 'roberta':
			self._tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
		elif tokenizer_model.lower() == 'bert':
			self._tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
		else:
			raise ValueError('tokenizer_model must either be roberta or bert')

		self._label_dict = {'entailment': 0, 'neutral': 1, 'contradiction': 2}

	@overrides
	def _read(self, file_path: str):
		for line in Reader(open(file_path)):
			yield self.text_to_instance(**line)

	@overrides
	def text_to_instance(self, 
						 premise: str, 
						 hypothesis: str, 
						 label: str = None, 
						 tag = None) -> Instance:
		
		# Ensure our inputs are properly contained in lists so that we can batch them
		premise_tokens = self._tokenizer.tokenize(premise)
		hypothesis_tokens = self._tokenizer.tokenize(hypothesis)
		premise_ids = self._tokenizer.convert_tokens_to_ids(premise_tokens)
		hypothesis_ids = self._tokenizer.convert_tokens_to_ids(hypothesis_tokens)

		fields = {'premise_ids': 	ArrayField(np.array(premise_ids), dtype=np.int64), 
				  'hypothesis_ids':	ArrayField(np.array(hypothesis_ids), dtype=np.int64), 	
				  'metadata':		MetadataField({'premise': premise, 'hypothesis': hypothesis, 
				  								   'premise_tokens': premise_tokens, 'hypothesis_tokens': hypothesis_tokens,
				  								   'label': label, 'tag': tag})}

		if label is not None:
			fields['label'] = ArrayField(np.array(self._label_dict[label]), dtype=np.int64)

		return Instance(fields)
