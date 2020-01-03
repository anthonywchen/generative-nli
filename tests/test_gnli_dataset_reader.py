from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.iterators.basic_iterator import BasicIterator
from jsonlines import Reader
import logging
from pytorch_transformers import RobertaTokenizer
import random

from src.gnli_dataset_reader import GNLIDatasetReader

random.seed(0)

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class Tests(AllenNlpTestCase):
	##########################################
	### Test dataset reader functions in `src/dataset_reader.py`
	##########################################
	def test_gnli_dataset_reader(self):
		def check_dataset_reader(file):
			tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
			max_premise_length = 128
			max_hypothesis_length = 80
			reader = GNLIDatasetReader('roberta-large', 
									   lazy=True, 
									   max_premise_length=max_premise_length,
									   max_hypothesis_length=max_hypothesis_length,
									   percent_data=0.001)

			for instance in reader.read(file):
				src = instance['src'].array.tolist()
				src_length = instance['src_lengths'].array.tolist()[0]
				prev_output_tokens = instance['prev_output_tokens'].array.tolist()
				target = instance['target'].array.tolist()
				target_length = instance['target_lengths'].array.tolist()[0]

				premise_tokens = instance['metadata'].metadata['premise_tokens']
				hypothesis_tokens = instance['metadata'].metadata['hypothesis_tokens']

				# Test encoder and decoder inputs are at max length
				assert len(src) == max_premise_length
				assert len(target) == len(prev_output_tokens) == max_hypothesis_length

				assert len(premise_tokens) == src_length - 2
				assert len(hypothesis_tokens) == target_length - 2

				# Test conversion of tokens to ids for encoder input and decoder input and target
				assert tokenizer.convert_tokens_to_ids(['<s>'] + premise_tokens + ['</s>']) == src[:src_length]
				assert tokenizer.convert_tokens_to_ids(['</s>', '<s>'] + hypothesis_tokens) == prev_output_tokens[:target_length]
				assert tokenizer.convert_tokens_to_ids(['<s>'] + hypothesis_tokens + ['</s>']) == target[:target_length]

				# Check padding
				assert tokenizer.convert_tokens_to_ids(['<pad>']*(max_premise_length-src_length)) == src[src_length:]
				assert tokenizer.convert_tokens_to_ids(['<pad>']*(max_hypothesis_length-target_length)) == prev_output_tokens[target_length:]
				assert tokenizer.convert_tokens_to_ids(['<pad>']*(max_hypothesis_length-target_length)) == target[target_length:]

		check_dataset_reader('data/snli/train.jsonl')