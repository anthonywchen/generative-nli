from allennlp.common.testing import AllenNlpTestCase
from jsonlines import Reader
import logging
import random
import tqdm as tqdm 

from src.bert_dataset_reader import BertNLIDatasetReader

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Tests(AllenNlpTestCase):
	##########################################
	### Test dataset reader functions in `src/dataset_reader.py`
	##########################################
	def test_bert_dataset_reader(self):
		random.seed(0)

		def check_tokenizer_dataset(tokenizer, file):
			reader = BertNLIDatasetReader(tokenizer, lazy=True)
			for i, instance in enumerate(reader.read(file)):
				if i > 100:	
					break

				input_ids = instance['input_ids'].array
				token_type_ids = instance['token_type_ids'].array
				premise_tokens = instance['metadata'].metadata['premise_tokens']

				# Check that `token_type_ids` has been created correctly
				assert len(input_ids) == len(token_type_ids)
				assert token_type_ids[:len(premise_tokens)+2].all() == 0
				assert token_type_ids[len(premise_tokens)+2:].all() == 1

				# Check that label has been created
				assert reader._label_dict[instance['metadata'].metadata['label']] == instance['label'].array

		# SNLI
		check_tokenizer_dataset('bert-base-uncased', 'data/snli/train.jsonl')
		check_tokenizer_dataset('roberta-base', 'data/snli/train.jsonl')
		check_tokenizer_dataset('bert-large-uncased', 'data/snli/train.jsonl')
		check_tokenizer_dataset('roberta-large', 'data/snli/train.jsonl')

		# MNLI
		check_tokenizer_dataset('bert-base-uncased', 'data/mnli/train.jsonl')
		check_tokenizer_dataset('roberta-base', 'data/mnli/train.jsonl')
		check_tokenizer_dataset('bert-large-uncased', 'data/mnli/train.jsonl')
		check_tokenizer_dataset('roberta-large', 'data/mnli/train.jsonl')

		# ANLI
		check_tokenizer_dataset('bert-base-uncased', 'data/anli/test.jsonl')
		check_tokenizer_dataset('roberta-base', 'data/anli/test.jsonl')
		check_tokenizer_dataset('bert-large-uncased', 'data/anli/test.jsonl')
		check_tokenizer_dataset('roberta-large', 'data/anli/test.jsonl')
		