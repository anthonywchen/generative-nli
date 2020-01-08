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
				 max_seq_length=None,
				 lazy=False) -> None:
		super().__init__(lazy)
		assert percent_data > 0 and percent_data <= 1
		self.percent_data = percent_data
		self.max_seq_length = max_seq_length
		self.tokenizer_class = pretrained_model.split('-')[0].lower()
		
		if self.tokenizer_class == 'roberta':	
			self._tokenizer = RobertaTokenizer.from_pretrained(pretrained_model)
		elif self.tokenizer_class == 'bert':
			self._tokenizer = BertTokenizer.from_pretrained(pretrained_model, do_lower_case=True)
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

		if self.percent_data < 1:
			logger.info('Sampling lines...')
			lines = random.sample(lines, num_lines_to_use)

		# Create instances
		for line in lines:
			yield self.text_to_instance(**line)

	@overrides
	def text_to_instance(self, premise: str, hypothesis: str, label: str = None, tag=None) -> Instance:
		premise_tokens = self._tokenizer.tokenize(premise)
		hypothesis_tokens = self._tokenizer.tokenize(hypothesis)

		if self.max_seq_length != None:
			self._truncate_seq_pair(premise_tokens, hypothesis_tokens)

		premise_ids = self._tokenizer.convert_tokens_to_ids(premise_tokens)
		hypothesis_ids = self._tokenizer.convert_tokens_to_ids(hypothesis_tokens)

		input_ids = self._tokenizer.add_special_tokens_sentences_pair(premise_ids, hypothesis_ids)
		token_type_ids = self.get_token_type_ids(input_ids)
		attention_mask = [1]*len(input_ids)

		# Add padding if max_seq_length is defined
		if self.max_seq_length != None:
			padding = [0] * (self.max_seq_length - len(input_ids))
			input_ids += padding
			attention_mask += padding
			token_type_ids += padding

		metadata = {'premise': premise, 
					'hypothesis': hypothesis, 
					'premise_tokens': premise_tokens,
					'hypothesis_tokens': hypothesis_tokens,
					'label': label, 'tag': tag}

		fields = {'input_ids': ArrayField(np.array(input_ids), dtype=np.int64),
				  'token_type_ids': ArrayField(np.array(token_type_ids), dtype=np.int64),
				  'attention_mask': ArrayField(np.array(attention_mask), dtype=np.int64),
				  'metadata': MetadataField(metadata)}

		if label is not None:
			fields['label'] = ArrayField(np.array(self._label_dict[label]), dtype=np.int64)

		return Instance(fields)

	def _truncate_seq_pair(self, tokens_a, tokens_b):
		"""Truncates a sequence pair in place to the maximum length."""
		# This is a simple heuristic which will always truncate the longer sequence
		# one token at a time. This makes more sense than truncating an equal percent
		# of tokens from each, since if one sequence is very short then each token
		# that's truncated likely contains more information than a longer sequence.
		# This code is taken from the HuggingFace library.

		num_special_tokens = 3 if self.tokenizer_class == 'bert' else 4
		max_length = self.max_seq_length - num_special_tokens

		while True:
			total_length = len(tokens_a) + len(tokens_b)
			if total_length <= max_length:
				break
			if len(tokens_a) > len(tokens_b):
				tokens_a.pop()
			else:
				tokens_b.pop()

	def get_token_type_ids(self, ids):
		for pos, token_id in enumerate(ids):
			if token_id == self.sep_id:
				token_type_ids = [0]*(pos+1) + [1]*(len(ids)-pos-1)
				return token_type_ids
