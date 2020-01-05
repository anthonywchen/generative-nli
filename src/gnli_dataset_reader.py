from jsonlines import Reader
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

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DatasetReader.register("gnli")
class GNLIDatasetReader(DatasetReader):
	def __init__(self,
				 pretrained_model: str,
				 max_premise_length: int,
				 max_hypothesis_length: int,
				 percent_data: float = 1,
				 lazy: bool = False) -> None:
		super().__init__(lazy)
		assert percent_data > 0 and percent_data <= 1
		self.percent_data = percent_data
		self.max_premise_length = max_premise_length
		self.max_hypothesis_length = max_hypothesis_length

		self._tokenizer = GNLITokenizer.from_pretrained(pretrained_model)
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
		####################
		##### Tokenization and truncation
		####################
		premise_tokens = self._tokenizer.tokenize(premise.strip())
		hypothesis_tokens = self._tokenizer.tokenize(hypothesis.strip())
		premise_tokens, hypothesis_tokens = self._truncate_input(premise_tokens, hypothesis_tokens)

		####################
		##### Create ids for encoder inputs, decoder inputs and decoder targets 
		####################
		src = []
		src.append(self._tokenizer.add_special_tokens_single_sentence(self._tokenizer.convert_tokens_to_ids([self._tokenizer.entail_token]+premise_tokens)))
		src.append(self._tokenizer.add_special_tokens_single_sentence(self._tokenizer.convert_tokens_to_ids([self._tokenizer.neutral_token]+premise_tokens)))
		src.append(self._tokenizer.add_special_tokens_single_sentence(self._tokenizer.convert_tokens_to_ids([self._tokenizer.contradict_token]+premise_tokens)))
		src_length = len(src[0])

		# Targets of the decoder: [<s> A B C D E <\s>]
		target = self._tokenizer.add_special_tokens_single_sentence(self._tokenizer.convert_tokens_to_ids(hypothesis_tokens))
		# Inputs of the decoder:  [<\s> <s> A B C D E]
		prev_output_tokens = [self._tokenizer.eos_token_id] + target[:-1]
		target_length = len(target)

		####################
		##### Padding of the input 
		####################
		# Pad the premise ids (the source)
		if self.max_premise_length:
			encoder_padding = [self._tokenizer.pad_token_id]*(self.max_premise_length - src_length)
			src = [s + encoder_padding for s in src]

		# Pad the hypothesis ids (the target)
		if self.max_hypothesis_length:
			decoder_padding = [self._tokenizer.pad_token_id]*(self.max_hypothesis_length - target_length)
			target += decoder_padding
			prev_output_tokens += decoder_padding

		# Replicate the hypothesis and the `src_lengths` three times
		target = [target]*3
		prev_output_tokens = [prev_output_tokens]*3
		src_length = [src_length]*3

		####################
		##### Create instance
		####################
		metadata = {'premise': premise,
					'hypothesis': hypothesis,
					'premise_tokens': premise_tokens,
					'hypothesis_tokens': hypothesis_tokens,
					'label': label, 'tag': tag}

		fields = {'src':	 			ArrayField(np.array(src), dtype=np.int64),
				  'src_lengths': 		ArrayField(np.array(src_length), dtype=np.int64),
				  'prev_output_tokens': ArrayField(np.array(prev_output_tokens), dtype=np.int64),
				  'target': 			ArrayField(np.array(target), dtype=np.int64),
				  'target_lengths':		ArrayField(np.array(target_length), dtype=np.int64),
				  'metadata': 			MetadataField(metadata)}

		if label is not None:
			fields['label'] = ArrayField(np.array(self._label_dict[label]), dtype=np.int64)

		return Instance(fields)

	def _truncate_input(self, premise_tokens, hypothesis_tokens):
		if self.max_premise_length:
			# Account for [<s>] + label_token + premise_tokens + [</s>]
			max_premise_length = self.max_premise_length - 3
			premise_tokens = premise_tokens[:max_premise_length]

		if self.max_hypothesis_length:
			# Account for [<s>] + hypothesis_tokens + [</s>]
			max_hypothesis_length = self.max_hypothesis_length - 2
			hypothesis_tokens = hypothesis_tokens[:max_hypothesis_length]

		return premise_tokens, hypothesis_tokens
