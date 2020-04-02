from allennlp.common.testing import AllenNlpTestCase
from allennlp.common import Params
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.models.model import Model
from math import isclose
import os
from pprint import pprint
import random
import torch

from src.gnli import GNLI
from src.gnli_dataset_reader import GNLIDatasetReader

os.environ["CUDA_VISIBLE_DEVICES"]="0"

class Tests(AllenNlpTestCase):
	def set_seed(self):
		random.seed(0)
		torch.manual_seed(0)

	def test_one_batch(self):
		self.set_seed()
		config = Params.from_file('tests/sample_gnli_config.json')
		gnli = Model.from_params(params=config['model'])
		reader = DatasetReader.from_params(params=config['dataset_reader'])
		iterator = DataIterator.from_params(params=config['iterator'])

		for batch in iterator(reader.read('data/mnli/dev.jsonl')):
			output_dict = gnli(**batch)
			output_dict['loss'].backward()
			del output_dict['metadata']
			break
		pprint(output_dict)
		assert isclose(output_dict['loss'].item(), 3.1145, abs_tol=1e-3)

	# def test_one_batch_encoder_input_label(self):
	# 	self.set_seed()
	# 	config = Params.from_file('tests/sample_gnli_config.json')

	# 	config['model']['have_encoder_label'] = True
	# 	gnli = Model.from_params(params=config['model'])
		
	# 	reader = DatasetReader.from_params(params=config['dataset_reader'])
		
	# 	iterator = DataIterator.from_params(params=config['iterator'])

	# 	for batch in iterator(reader.read('data/mnli/dev.jsonl')):
	# 		output_dict = gnli(**batch)
	# 		output_dict['loss'].backward()
	# 		del output_dict['metadata']
	# 		break

	# 	assert isclose(output_dict['loss'].item(), 8.3758, abs_tol=1e-3)