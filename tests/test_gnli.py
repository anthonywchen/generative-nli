from allennlp.common.testing import AllenNlpTestCase
from allennlp.common import Params
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.data.iterators.basic_iterator import BasicIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
import json
import logging
from math import isclose
import os
from pprint import pprint
import random
import torch

from src.apex_trainer import ApexTrainer
from src.gnli_dataset_reader import GNLIDatasetReader
from src.gnli import GNLI

ABS_TOL = 0.000001
os.environ["CUDA_VISIBLE_DEVICES"]="0"		
vocab = Vocabulary()

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class Tests(AllenNlpTestCase):
	def set_seed(self):
		random.seed(0)
		torch.manual_seed(0)

	def test_embeddings(self):
		config = Params.from_file('tests/sample_gnli_config.json')
		gnli = Model.from_params(params=config['model'])
		bart = torch.hub.load('pytorch/fairseq', 'bart.large').model

		# Check that GNLI encoder and decoder are tied
		assert gnli._bart.encoder.embed_tokens == gnli._bart.decoder.embed_tokens

		# Check that GNLI and original BART token embeddings values match
		assert torch.all(list(gnli._bart.encoder.embed_tokens.parameters())[0][:-3] == list(bart.encoder.embed_tokens.parameters())[0]).item()

	def test_one_batch(self):
		self.set_seed()
		config = Params.from_file('tests/sample_gnli_config.json')
		config['model']['replace_bos_token'] = False
		config['model']['discriminative_loss_weight'] = 1
		gnli = Model.from_params(params=config['model'])

		reader_params = config.pop('dataset_reader')
		reader = DatasetReader.by_name(reader_params.pop('type')).from_params(reader_params)

		iterator_params = config.pop('iterator')
		iterator_params.params["batch_size"] = 3
		iterator = DataIterator.by_name(iterator_params.pop('type')).from_params(iterator_params)

		instances = reader.read('data/mnli/dev.jsonl')

		for batch in iterator(instances):
			output_dict = gnli(**batch)
			output_dict['loss'].backward()
			break
		del output_dict['metadata']
		assert isclose(output_dict['loss'].item(), 8.3528, abs_tol=1e-3)

	# def test_one_batch_replace_eos(self):
	# 	self.set_seed()
	# 	config = Params.from_file('tests/sample_gnli_config.json')
	# 	config['model']['replace_bos_token'] = True
	# 	config['model']['discriminative_loss_weight'] = 1
	# 	gnli = Model.from_params(params=config['model'])

	# 	reader_params = config.pop('dataset_reader')
	# 	reader = DatasetReader.by_name(reader_params.pop('type')).from_params(reader_params)

	# 	iterator_params = config.pop('iterator')
	# 	iterator_params.params["batch_size"] = 3
	# 	iterator = DataIterator.by_name(iterator_params.pop('type')).from_params(iterator_params)

	# 	instances = reader.read('data/mnli/dev.jsonl')

	# 	for batch in iterator(instances):
	# 		output_dict = gnli(**batch)
	# 		output_dict['loss'].backward()
	# 		break
	# 	del output_dict['metadata']
	# 	assert isclose(output_dict['loss'].item(), 7.1976, abs_tol=1e-3)