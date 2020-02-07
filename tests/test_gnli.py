from allennlp.common.testing import AllenNlpTestCase
from allennlp.common import Params
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.data.iterators.basic_iterator import BasicIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.training.trainer import TrainerBase
import json
import logging
from math import isclose
import os
from os.path import isdir, join
import random
import shutil
import torch

from src.apex_trainer import ApexTrainer
from src.gnli_dataset_reader import GNLIDatasetReader
from src.gnli import GNLI

ABS_TOL = 0.000001
os.environ["CUDA_VISIBLE_DEVICES"]="2"		
vocab = Vocabulary()

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class Tests(AllenNlpTestCase):
	def set_seed(self):
		random.seed(0)
		torch.manual_seed(0)

	def test_gnli_training_dual_loss(self):
		""" 
		Tests that we can run a training run, and 
		record the output scores so that we can always check back 
		"""
		print('\ntest_gnli\n')
		self.set_seed()

		config = Params.from_file('tests/sample_gnli_config.json')
		config.params['model']['discriminative_loss_weight'] = .8
		output_directory = 'tests/gnli'
		if isdir(output_directory): 
			shutil.rmtree(output_directory)

		trainer = ApexTrainer.from_params(params=config, serialization_dir=output_directory)
		trainer.train()

		# Check that final metrics are correct. This can be useful since different versions 
		# sometimes yield different results and this can potentially reveal this discrepency. 
		final_metrics = json.load(open(join(output_directory, 'metrics_epoch_19.json')))
		assert isclose(final_metrics['training_accuracy'], 0.7872727272727272, abs_tol=ABS_TOL)
		assert isclose(final_metrics['training_loss'], 1.0372580289840698, abs_tol=ABS_TOL)
		assert isclose(final_metrics['best_validation_accuracy'], 0.7, abs_tol=ABS_TOL)
		assert isclose(final_metrics['best_validation_loss'], 1.4115259647369385, abs_tol=ABS_TOL)
		assert isclose(final_metrics['best_epoch'], 17)

	def test_gnli_training_half_prec(self):
		""" 
		Tests that we can run a training run, and 
		record the output scores so that we can always check back 
		"""
		print('\ntest_gnli_dual\n')
		self.set_seed()

		config = Params.from_file('tests/sample_gnli_config.json')
		config.params['model']['discriminative_loss_weight'] = .5
		config.params['trainer']['half_precision'] = True
		config.params['trainer']['opt_level'] = 'O2'
		output_directory = 'tests/gnli_half_prec'
		if isdir(output_directory): 
			shutil.rmtree(output_directory)

		trainer = ApexTrainer.from_params(params=config, serialization_dir=output_directory)
		trainer.train()

		final_metrics = json.load(open(join(output_directory, 'metrics_epoch_19.json')))
		assert isclose(final_metrics['training_accuracy'], 0.5272727272727272, abs_tol=ABS_TOL)
		assert isclose(final_metrics['training_loss'], 1.4289642708642143, abs_tol=ABS_TOL)
		assert isclose(final_metrics['best_validation_accuracy'], 0.8, abs_tol=ABS_TOL)
		assert isclose(final_metrics['best_validation_loss'], 1.4033045768737793, abs_tol=ABS_TOL)
		assert isclose(final_metrics['best_epoch'], 17)