import logging
import numpy as np
from overrides import overrides
import random
from typing import Callable, Dict, Iterable, Iterator, List

from allennlp.common import Params, Tqdm
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import ArrayField
from allennlp.data.fields.metadata_field import MetadataField
from allennlp.data.instance import Instance

from src.gnli_tokenizer import GNLITokenizer
from src import utils 

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DatasetReader.register("gnli")
class GNLIDatasetReader(DatasetReader):
	def __init__(self,
				 pretrained_model: str,
				 max_premise_length: int = None,
				 max_hypothesis_length: int = None,
				 percent_data: float = 1,
				 lazy: bool = False) -> None:
		super().__init__(lazy)
		assert 0 < percent_data <= 1
		self.percent_data = percent_data
		self.max_premise_length = max_premise_length
		self.max_hypothesis_length = max_hypothesis_length

		self._tokenizer = GNLITokenizer.from_pretrained(pretrained_model)
		self._label_dict = {'entailment': 0, 'neutral': 1, 'contradiction': 2}

	@overrides
	def _read(self, file_path: str):
		lines = utils.read_data(file_path, self.percent_data)

		# Create instances
		for line in lines:
			yield self.text_to_instance(**line)

	@overrides
	def text_to_instance(self, premise: str, hypothesis: str, label: str = None, tag=None) -> Instance:
		####################
		##### Tokenization and truncation
		####################
		premise_tokens = self._tokenizer.tokenize(premise.strip())
		hypothesis_tokens = self._tokenizer.tokenize(hypothesis.strip())
		premise_tokens, hypothesis_tokens = self._truncate_input(premise_tokens, hypothesis_tokens)

		# Input of the encoder: [<s> A B C D E </s>]
		src = self._tokenizer.add_special_tokens_single_sentence(self._tokenizer.convert_tokens_to_ids(premise_tokens))
		
		# Targets of the decoder: [V W X Y Z <\s>]
		target = self._tokenizer.add_special_tokens_single_sentence(self._tokenizer.convert_tokens_to_ids(hypothesis_tokens))[1:]

		# Inputs of the decoder:  [<s> V W X Y Z]
		prev_output_tokens = [self._tokenizer.bos_token_id] + target[:-1]

		metadata = {'premise': premise,
					'hypothesis': hypothesis,
					'premise_tokens': premise_tokens,
					'hypothesis_tokens': hypothesis_tokens,
					'label': label, 
					'tag': tag}

		fields = {'src':	 			ArrayField(np.array(src), dtype=np.int64, padding_value=1),
				  'prev_output_tokens': ArrayField(np.array(prev_output_tokens), dtype=np.int64, padding_value=1),
				  'target': 			ArrayField(np.array(target), dtype=np.int64, padding_value=1),
				  'src_lengths': 		ArrayField(np.array(len(src)), dtype=np.int64),
				  'target_lengths':		ArrayField(np.array(len(target)), dtype=np.int64),
				  'metadata': 			MetadataField(metadata)}

		if label is not None:
			fields['label'] = ArrayField(np.array(self._label_dict[label]), dtype=np.int64)

		return Instance(fields)

	def _truncate_input(self, premise_tokens, hypothesis_tokens):
		if self.max_premise_length:
			# Account for `[<s>] + premise_tokens + [</s>]`
			max_premise_length = self.max_premise_length - 2
			premise_tokens = premise_tokens[:max_premise_length]

		if self.max_hypothesis_length:
			# Account for `hypothesis_tokens + [</s>]`
			max_hypothesis_length = self.max_hypothesis_length - 1
			hypothesis_tokens = hypothesis_tokens[:max_hypothesis_length]

		return premise_tokens, hypothesis_tokens
