from jsonlines import Reader
import logging
import numpy as np
from overrides import overrides
from pytorch_transformers import BertTokenizer, RobertaTokenizer
import random
from typing import Callable, Dict, Iterable, Iterator, List

from allennlp.common import Params, Tqdm
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import ArrayField
from allennlp.data.fields.metadata_field import MetadataField
from allennlp.data.instance import Instance

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DatasetReader.register("bertnli")
class BertNLIDatasetReader(DatasetReader):
	def __init__(self,
				 pretrained_model,
				 percent_data=1,
				 lazy=False,
				 shuffle=False) -> None:
		super().__init__(lazy)
		assert percent_data > 0 and percent_data <= 1
		self.percent_data = percent_data
		self.shuffle = shuffle

		self.tokenizer_class = pretrained_model.split('-')[0].lower()

		if self.tokenizer_class == 'roberta':	
			self._tokenizer = RobertaTokenizer.from_pretrained(pretrained_model)
		elif self.tokenizer_class == 'bert':
			self._tokenizer = BertTokenizer.from_pretrained(pretrained_model)
		else:									
			raise ValueError('tokenizer_model must either be roberta or bert')

		self.sep_id = self._tokenizer.encode(self._tokenizer.sep_token)[0]
		self._label_dict = {'entailment': 0, 'neutral': 1, 'contradiction': 2}

	@overrides
	def _read(self, file_path: str):
		# Load in all lines
		with open(file_path) as f:
			lines = [line for line in Reader(f)]

		# Determine how many lines we will use as a percent of the data
		num_lines_to_use = int(len(lines)*self.percent_data)
		logger.info('Number of data points: %d', num_lines_to_use)

		if self.shuffle:
			logger.info('Shuffling...')
			random.shuffle(lines)

		# Create instances
		for line in lines[:num_lines_to_use]:
			yield self.text_to_instance(**line)

	@overrides
	def text_to_instance(self, premise: str, hypothesis: str, label: str = None, tag=None) -> Instance:
		premise_ids = self._tokenizer.encode(premise)
		hypothesis_ids = self._tokenizer.encode(hypothesis)
		input_ids = self._tokenizer.add_special_tokens_sentences_pair(premise_ids, hypothesis_ids)
		token_type_ids = self.get_token_type_ids(input_ids)
		attention_mask = [1]*len(input_ids)

		metadata = {'premise': premise, 'hypothesis': hypothesis, 
					'premise_tokens': self._tokenizer.tokenize(premise), 
					'hypothesis_tokens': self._tokenizer.tokenize(hypothesis),
					'label': label, 'tag': tag}

		fields = {'input_ids': ArrayField(np.array(input_ids), dtype=np.int64),
				  'token_type_ids': ArrayField(np.array(token_type_ids), dtype=np.int64),
				  'attention_mask': ArrayField(np.array(attention_mask), dtype=np.int64),
				  'metadata': MetadataField(metadata)}

		if label is not None:
			fields['label'] = ArrayField(np.array(self._label_dict[label]), dtype=np.int64)

		return Instance(fields)

	def get_token_type_ids(self, ids):
		for pos, token_id in enumerate(ids):
			if token_id == self.sep_id:
				token_type_ids = [0]*(pos+1) + [1]*(len(ids)-pos-1)
				return token_type_ids
