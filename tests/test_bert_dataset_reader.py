from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.iterators.basic_iterator import BasicIterator
from jsonlines import Reader
import logging
from pytorch_transformers import BertTokenizer, RobertaTokenizer
import random
import torch

from src.bert_dataset_reader import BertNLIDatasetReader

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class Tests(AllenNlpTestCase):
	##########################################
	### Test dataset reader functions in `src/dataset_reader.py`
	##########################################
	def test_bert_dataset_reader(self):
		random.seed(0)

		def check_tokenizer_dataset(pretrained_model, file):
			name = pretrained_model.split('-')[0].lower()
			assert name in ['bert', 'roberta']
			tokenizer = RobertaTokenizer.from_pretrained(pretrained_model) if name=='roberta' else  BertTokenizer.from_pretrained(pretrained_model)
			reader = BertNLIDatasetReader(pretrained_model, lazy=True, percent_data=0.001)

			for instance in reader.read(file):
				input_ids = instance['input_ids'].array.tolist()
				token_type_ids = instance['token_type_ids'].array
				attention_mask = instance['attention_mask'].array
				premise_tokens = instance['metadata'].metadata['premise_tokens']
				hypothesis_tokens = instance['metadata'].metadata['hypothesis_tokens']

				assert len(input_ids) == len(token_type_ids) == len(attention_mask)
				assert attention_mask.all() == 1
				assert reader._label_dict[instance['metadata'].metadata['label']] == instance['label'].array

				if name == 'bert':
					tokenizer.convert_tokens_to_ids(['[CLS]'] + premise_tokens + ['[SEP]'] + hypothesis_tokens + ['[SEP]']) == input_ids
					segment_divide = len(premise_tokens)+2
					assert input_ids[:segment_divide][-1] == 102
					assert token_type_ids[:segment_divide].all() == 0
					assert token_type_ids[segment_divide:].all() == 1
				elif name == 'roberta':
					assert tokenizer.convert_tokens_to_ids(['<s>'] + premise_tokens + ['</s>', '</s>'] + hypothesis_tokens + ['</s>']) == input_ids

		# SNLI
		check_tokenizer_dataset('bert-base-uncased', 'data/snli/train.jsonl')
		check_tokenizer_dataset('roberta-base', 'data/snli/train.jsonl')

		# MNLI
		check_tokenizer_dataset('bert-base-uncased', 'data/mnli/train.jsonl')
		check_tokenizer_dataset('roberta-base', 'data/mnli/train.jsonl')
		
	##########################################
	### Test dataset iterator
	##########################################
	def test_bert_iterator(self):
		random.seed(0)

		def check_iterator(pretrained_model, file, max_seq_length=None):
			batch_size = 10
			name = pretrained_model.split('-')[0].lower()
			tokenizer = RobertaTokenizer.from_pretrained(pretrained_model) if name=='roberta' else  BertTokenizer.from_pretrained(pretrained_model)
			reader = BertNLIDatasetReader(pretrained_model, lazy=True, percent_data=0.001, max_seq_length=max_seq_length)
			iterator = BasicIterator(batch_size=batch_size, max_instances_in_memory=10000)

			for batch_dict in iterator(reader.read(file), num_epochs=1):
				assert batch_dict['input_ids'].size() == batch_dict['token_type_ids'].size() == batch_dict['attention_mask'].size()

				for idx in range(batch_dict['input_ids'].size(0)):
					input_ids = batch_dict['input_ids'][idx].numpy().tolist()
					token_type_ids = batch_dict['token_type_ids'][idx]
					attention_mask = batch_dict['attention_mask'][idx]
					premise = batch_dict['metadata'][idx]['premise_tokens']
					hypothesis = batch_dict['metadata'][idx]['hypothesis_tokens']

					num_extra_tokens = 3 if name == 'bert' else 4
					num_input_ids = len(premise) + len(hypothesis) + num_extra_tokens
					
					# Check input ids
					if name == 'bert':
						assert input_ids[:num_input_ids] == tokenizer.convert_tokens_to_ids(['[CLS]'] + premise + ['[SEP]'] + hypothesis + ['[SEP]'])
						
						segment_divide = len(premise)+2
						assert input_ids[:segment_divide][-1] == 102
						assert torch.sum(token_type_ids[:segment_divide]) == 0
						assert torch.sum(token_type_ids[segment_divide:num_input_ids]) == num_input_ids-segment_divide
						assert torch.sum(token_type_ids[num_input_ids:]) == 0
					else:
						assert input_ids[:num_input_ids] == tokenizer.convert_tokens_to_ids(['<s>'] + premise + ['</s>']*2 + hypothesis + ['</s>'])

					# Check attention mask
					assert torch.sum(attention_mask[:num_input_ids]).item() == num_input_ids
					assert torch.sum(attention_mask[num_input_ids:]).item() == 0

		# SNLI
		check_iterator('bert-base-uncased', 'data/snli/train.jsonl')
		check_iterator('roberta-base', 'data/snli/train.jsonl')

		# MNLI
		check_iterator('bert-base-uncased', 'data/mnli/train.jsonl')
		check_iterator('roberta-base', 'data/mnli/train.jsonl')
		check_iterator('bert-base-uncased', 'data/mnli/train.jsonl', max_seq_length=128)
		check_iterator('roberta-base', 'data/mnli/train.jsonl', max_seq_length=128)