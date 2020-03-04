from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.iterators.basic_iterator import BasicIterator
from jsonlines import Reader
import logging
import random
import torch
from tqdm import tqdm

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
		gnli_tokenizer = GNLITokenizer.from_pretrained('roberta-large')
		bart = torch.hub.load('pytorch/fairseq', 'bart.large')

		for line in tqdm(Reader(open('data/mnli/dev.jsonl'))):
			premise = line['premise'].strip()
			hypothesis = line['hypothesis'].strip()
			assert gnli_tokenizer.add_special_tokens_single_sentence(gnli_tokenizer.encode(premise)) == bart.encode(' ' + premise).tolist()
			assert gnli_tokenizer.add_special_tokens_single_sentence(gnli_tokenizer.encode(hypothesis)) == bart.encode(' ' + hypothesis).tolist()

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
		reader = GNLIDatasetReader('roberta-large', max_premise_length=max_premise_length, max_hypothesis_length=max_hypothesis_length)

		for instance in reader.read('data/mnli/dev.jsonl'):
			target = instance['target'].array.tolist()
			premise_tokens = instance['metadata'].metadata['premise_tokens']
			hypothesis_tokens = instance['metadata'].metadata['hypothesis_tokens']
			target_length = instance['target_lengths'].array.item()
			assert reader._label_dict[instance['metadata'].metadata['label']] == instance['label'].array

			src = instance['src'].array.tolist()
			prev_output_tokens = instance['prev_output_tokens'].array.tolist()
			src_length = instance['src_lengths'].array.tolist()

			# Test encoder and decoder inputs are at max length
			assert len(src) == max_premise_length
			assert len(target) == len(prev_output_tokens) == max_hypothesis_length
			assert len(premise_tokens) == src_length - 2
			assert len(hypothesis_tokens) == target_length - 1

			assert tokenizer.convert_tokens_to_ids(['<s>'] + premise_tokens + ['</s>']) == src[:src_length]
			assert tokenizer.convert_tokens_to_ids(['<s>'] + hypothesis_tokens) == prev_output_tokens[:target_length]
			assert tokenizer.convert_tokens_to_ids(hypothesis_tokens + ['</s>']) == target[:target_length]
			assert target[:target_length] == prev_output_tokens[1:target_length] + [tokenizer.eos_token_id]

			# Check padding
			assert tokenizer.convert_tokens_to_ids(['<pad>']*(max_premise_length-src_length)) == src[src_length:]
			assert tokenizer.convert_tokens_to_ids(['<pad>']*(max_hypothesis_length-target_length)) == prev_output_tokens[target_length:]
			assert tokenizer.convert_tokens_to_ids(['<pad>']*(max_hypothesis_length-target_length)) == target[target_length:]

	def test_gnli_dataset_reader_no_padding(self):
		tokenizer = GNLITokenizer.from_pretrained('roberta-large')
		reader = GNLIDatasetReader('roberta-large')

		for instance in reader.read('data/mnli/dev.jsonl'):
			target = instance['target'].array.tolist()
			premise_tokens = instance['metadata'].metadata['premise_tokens']
			hypothesis_tokens = instance['metadata'].metadata['hypothesis_tokens']
			target_length = instance['target_lengths'].array.item()
			assert reader._label_dict[instance['metadata'].metadata['label']] == instance['label'].array

			src = instance['src'].array.tolist()
			prev_output_tokens = instance['prev_output_tokens'].array.tolist()
			src_length = instance['src_lengths'].array.tolist()

			# Test encoder and decoder inputs are at max length
			assert len(premise_tokens) + 2 == src_length == len(src)
			assert len(hypothesis_tokens) + 1 == target_length == len(target) == len(prev_output_tokens)

			assert tokenizer.convert_tokens_to_ids(['<s>'] + premise_tokens + ['</s>']) == src
			assert tokenizer.convert_tokens_to_ids(['<s>'] + hypothesis_tokens) == prev_output_tokens
			assert tokenizer.convert_tokens_to_ids(hypothesis_tokens + ['</s>']) == target
			assert target[:target_length] == prev_output_tokens[1:target_length] + [tokenizer.eos_token_id]