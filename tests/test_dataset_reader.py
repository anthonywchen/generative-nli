from allennlp.common.testing import AllenNlpTestCase
from jsonlines import Reader
from pytorch_transformers import BertTokenizer, RobertaTokenizer
import random
import tqdm as tqdm 

from src.dataset_reader import NLIDatasetReader
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class Tests(AllenNlpTestCase):
	##########################################
	### Test dataset reader functions in `src/dataset_reader.py`
	##########################################
	def test_dataset_reader(self):
		random.seed(0)

		def check_tokenizer_dataset(tokenizer, file):
			reader = NLIDatasetReader(1, 1, tokenizer, lazy=True)
			for i, instance in enumerate(reader.read(file)):
				if i > 100:
					break

				premise = instance['metadata'].metadata['premise']
				hypothesis = instance['metadata'].metadata['hypothesis']
				premise_tokens = instance['metadata'].metadata['premise_tokens']
				hypothesis_tokens = instance['metadata'].metadata['hypothesis_tokens']
				premise_ids = instance['premise_ids'].array
				hypothesis_ids = instance['hypothesis_ids'].array

				assert reader._tokenizer.tokenize(premise) == premise_tokens
				assert reader._tokenizer.tokenize(hypothesis) == hypothesis_tokens
				assert reader._tokenizer.convert_ids_to_tokens(premise_ids) == premise_tokens
				assert reader._tokenizer.convert_ids_to_tokens(hypothesis_ids) == hypothesis_tokens
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
		