from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.iterators.basic_iterator import BasicIterator
from jsonlines import Reader
import logging
from pytorch_transformers import RobertaTokenizer
import random

from src.gnli_dataset_reader import GNLIDatasetReader
from src.gnli_tokenizer import GNLITokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
random.seed(0)

class Tests(AllenNlpTestCase):
	##########################################
	### Test dataset reader functions in `src/dataset_reader.py`
	##########################################
	def test_gnli_tokenizer(self):
		""" Tests that gnli tokenizer matches the roberta tokenizer"""
		roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
		gnli_tokenizer = GNLITokenizer.from_pretrained('roberta-large')

		for line in Reader(open('data/mnli/dev.jsonl')):
			premise = line['premise'].strip()
			assert roberta_tokenizer.tokenize(premise) == gnli_tokenizer.tokenize(premise)
			assert roberta_tokenizer.tokenize(premise) == gnli_tokenizer.tokenize(premise)

		# Test that the label tokens and ids were set correctly
		entail_token = gnli_tokenizer.entail_token
		entail_token_id = gnli_tokenizer.entail_token_id
		neutral_token = gnli_tokenizer.neutral_token
		neutral_token_id = gnli_tokenizer.neutral_token_id
		contradict_token = gnli_tokenizer.contradict_token
		contradict_token_id = gnli_tokenizer.contradict_token_id

		assert entail_token == '<entailment>'
		assert entail_token_id == 50265
		assert neutral_token == '<neutral>'
		assert neutral_token_id == 50266
		assert contradict_token == '<contradiction>'
		assert contradict_token_id == 50267

		# Test that the conversion functions work as expected
		assert gnli_tokenizer.convert_tokens_to_ids([entail_token, neutral_token, contradict_token]) == \
			[entail_token_id, neutral_token_id, contradict_token_id]
 
	def test_gnli_dataset_reader(self):
		tokenizer = GNLITokenizer.from_pretrained('roberta-large')
		max_premise_length = 128
		max_hypothesis_length = 80
		reader = GNLIDatasetReader('roberta-large', max_premise_length=max_premise_length,
								   max_hypothesis_length=max_hypothesis_length, percent_data=0.001)

		for instance in reader.read('data/mnli/train.jsonl'):
			premise_tokens = instance['metadata'].metadata['premise_tokens']
			hypothesis_tokens = instance['metadata'].metadata['hypothesis_tokens']
			target_length = instance['target_lengths'].array
			assert reader._label_dict[instance['metadata'].metadata['label']] == instance['label'].array

			# Iterate through the labels
			for i, cur_class in zip(range(3), reader._label_dict) :
				src = instance['src'].array.tolist()[i]
				prev_output_tokens = instance['prev_output_tokens'].array.tolist()[i]
				target = instance['target'].array.tolist()[i]
				src_length = instance['src_lengths'].array.tolist()[i]

				# Test encoder and decoder inputs are at max length
				assert len(src) == max_premise_length
				assert len(target) == len(prev_output_tokens) == max_hypothesis_length

				assert len(premise_tokens) == src_length - 3
				assert len(hypothesis_tokens) == target_length - 2

				# Test conversion of tokens to ids for encoder input and decoder input and target
				cur_class_token = '<'+cur_class+'>'
				assert tokenizer.convert_tokens_to_ids(['<s>'] + [cur_class_token] + premise_tokens + ['</s>']) == src[:src_length]
				assert tokenizer.convert_tokens_to_ids(['</s>', '<s>'] + hypothesis_tokens) == prev_output_tokens[:target_length]
				assert tokenizer.convert_tokens_to_ids(['<s>'] + hypothesis_tokens + ['</s>']) == target[:target_length]

				# Check padding
				assert tokenizer.convert_tokens_to_ids(['<pad>']*(max_premise_length-src_length)) == src[src_length:]
				assert tokenizer.convert_tokens_to_ids(['<pad>']*(max_hypothesis_length-target_length)) == prev_output_tokens[target_length:]
				assert tokenizer.convert_tokens_to_ids(['<pad>']*(max_hypothesis_length-target_length)) == target[target_length:]