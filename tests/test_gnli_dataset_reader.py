from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.iterators.basic_iterator import BasicIterator
from jsonlines import Reader
import logging
from pytorch_transformers import RobertaTokenizer
import random
import torch
from tqdm import tqdm

from src.gnli_dataset_reader import GNLIDatasetReader

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
random.seed(0)

class Tests(AllenNlpTestCase):
	##########################################
	### Test dataset reader functions in `src/dataset_reader.py`
	##########################################
	def test_gnli_tokenizer(self):
		""" Tests that gnli tokenizer matches the roberta tokenizer"""
		roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
		bart = torch.hub.load('pytorch/fairseq', 'bart.large')

		for line in tqdm(Reader(open('data/mnli/dev.jsonl'))):
			premise = line['premise'].strip()
			hypothesis = line['hypothesis'].strip()
			assert roberta_tokenizer.add_special_tokens_single_sentence(roberta_tokenizer.encode(premise)) == bart.encode(' ' + premise).tolist()
			assert roberta_tokenizer.add_special_tokens_single_sentence(roberta_tokenizer.encode(hypothesis)) == bart.encode(' ' + hypothesis).tolist()

	def test_gnli_dataset_reader(self):
		tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
		max_premise_length = 128
		max_hypothesis_length = 80
		reader = GNLIDatasetReader('roberta-large', max_premise_length=max_premise_length, max_hypothesis_length=max_hypothesis_length)

		for instance in reader.read('data/mnli/dev.jsonl'):
			target = instance['target'].array.tolist()
			premise_tokens = instance['metadata'].metadata['premise_tokens']
			hypothesis_tokens = instance['metadata'].metadata['hypothesis_tokens']
			target_length = instance['target_lengths'].array.item()
			assert reader._label_dict[instance['metadata'].metadata['label']] == instance['label'].array

			# Iterate through the labels
			for i, cur_class in zip(range(3), reader._label_dict) :
				src = instance['src'].array.tolist()[i]
				prev_output_tokens = instance['prev_output_tokens'].array.tolist()[i]
				src_length = instance['src_lengths'].array.tolist()[i]

				# Test encoder and decoder inputs are at max length
				assert len(src) == max_premise_length
				assert len(target) == len(prev_output_tokens) == max_hypothesis_length
				assert len(premise_tokens) == src_length - 2
				assert len(hypothesis_tokens) == target_length - 2

				# Test conversion of tokens to ids for encoder input and decoder input and target
				assert tokenizer.convert_tokens_to_ids(['<s>'] + premise_tokens + ['</s>']) == src[:src_length]
				assert tokenizer.convert_tokens_to_ids(['</s>', '<s>'] + hypothesis_tokens) == prev_output_tokens[:target_length]
				assert tokenizer.convert_tokens_to_ids(['<s>'] + hypothesis_tokens + ['</s>']) == target[:target_length]
				assert target[:target_length] == prev_output_tokens[1:target_length] + [tokenizer.eos_token_id]

				# Check padding
				assert tokenizer.convert_tokens_to_ids(['<pad>']*(max_premise_length-src_length)) == src[src_length:]
				assert tokenizer.convert_tokens_to_ids(['<pad>']*(max_hypothesis_length-target_length)) == prev_output_tokens[target_length:]
				assert tokenizer.convert_tokens_to_ids(['<pad>']*(max_hypothesis_length-target_length)) == target[target_length:]

	def test_gnli_dataset_reader_no_padding(self):
		tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
		reader = GNLIDatasetReader('roberta-large')

		for instance in reader.read('data/mnli/dev.jsonl'):
			target = instance['target'].array.tolist()
			premise_tokens = instance['metadata'].metadata['premise_tokens']
			hypothesis_tokens = instance['metadata'].metadata['hypothesis_tokens']
			target_length = instance['target_lengths'].array.item()
			assert reader._label_dict[instance['metadata'].metadata['label']] == instance['label'].array

			# Iterate through the labels
			for i, cur_class in zip(range(3), reader._label_dict) :
				src = instance['src'].array.tolist()[i]
				prev_output_tokens = instance['prev_output_tokens'].array.tolist()[i]
				src_length = instance['src_lengths'].array.tolist()[i]

				# Test encoder and decoder inputs are at max length
				assert len(premise_tokens) + 2 == src_length == len(src)
				assert len(hypothesis_tokens) + 2 == target_length == len(target) == len(prev_output_tokens)

				# Test conversion of tokens to ids for encoder input and decoder input and target
				assert tokenizer.convert_tokens_to_ids(['<s>'] + premise_tokens + ['</s>']) == src
				assert tokenizer.convert_tokens_to_ids(['</s>', '<s>'] + hypothesis_tokens) == prev_output_tokens
				assert tokenizer.convert_tokens_to_ids(['<s>'] + hypothesis_tokens + ['</s>']) == target
				assert target[:target_length] == prev_output_tokens[1:target_length] + [tokenizer.eos_token_id]
