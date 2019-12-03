from allennlp.common.testing import AllenNlpTestCase
from allennlp.common import Params
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.data.iterators.basic_iterator import BasicIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.training.trainer import TrainerBase
from copy import deepcopy
import json
import logging
from math import isclose
import os
from os.path import isdir, join
import random
import shutil
import torch

from src.bert_dataset_reader import BertNLIDatasetReader
from src.bert import BertNLI

ABS_TOL = 0.000001
os.environ["CUDA_VISIBLE_DEVICES"]="1"		
vocab = Vocabulary()

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class Tests(AllenNlpTestCase):
	def set_seed(self):
		random.seed(0)
		torch.manual_seed(0)

	def test_feed_one_batch(self):
		""" Tests if we can feed in batches of data through our model """
		print('\nTest feeding in one batch through our model\n')
		self.set_seed()

		config = Params.from_file('tests/sample_bert_base_config.json')
		model = Model.from_params(vocab=vocab, params=config['model'])
		
		reader_params = config.pop('dataset_reader')
		reader = DatasetReader.by_name(reader_params.pop('type')).from_params(reader_params)

		iterator_params = config.pop('iterator')
		iterator = DataIterator.by_name(iterator_params.pop('type')).from_params(iterator_params)

		instances = reader.read('data/snli/train.jsonl')

		for batch in iterator(instances):
			output_dict = model(**batch)
			output_dict['loss'].backward()
			break

	def test_bert_base_training(self):
		""" 
		Tests that we can run a training run, and 
		record the output scores so that we can always check back 
		"""
		print('\ntest_bert_base_training\n')
		self.set_seed()

		config = Params.from_file('tests/sample_bert_base_config.json')
		output_directory = 'tests/bert_base'
		if isdir(output_directory): 
			shutil.rmtree(output_directory)

		# Test training runs smoothly
		trainer = TrainerBase.from_params(params=config, serialization_dir=output_directory)
		trainer.train()

		# Check that final metrics are correct. This can be useful since different versions 
		# sometimes yield different results and this can potentially reveal this discrepency. 
		final_metrics = json.load(open(join(output_directory, 'metrics_epoch_4.json')))
		assert isclose(final_metrics['training_accuracy'], 0.8418181818181818, abs_tol=ABS_TOL)
		assert isclose(final_metrics['best_validation_accuracy'], 0.8, abs_tol=ABS_TOL)
		assert isclose(final_metrics['training_loss'], 0.722174068291982, abs_tol=ABS_TOL)
		assert isclose(final_metrics['best_validation_loss'], 0.6586655676364899, abs_tol=ABS_TOL)