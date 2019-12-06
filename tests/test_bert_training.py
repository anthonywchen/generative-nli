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

		trainer = TrainerBase.from_params(params=config, serialization_dir=output_directory)
		trainer.train()

		# Check that final metrics are correct. This can be useful since different versions 
		# sometimes yield different results and this can potentially reveal this discrepency. 
		final_metrics = json.load(open(join(output_directory, 'metrics_epoch_4.json')))
		assert isclose(final_metrics['training_accuracy'], 0.8418181818181818, abs_tol=ABS_TOL)
		assert isclose(final_metrics['best_validation_accuracy'], 0.8, abs_tol=ABS_TOL)
		assert isclose(final_metrics['training_loss'], 0.722174068291982, abs_tol=ABS_TOL)
		assert isclose(final_metrics['best_validation_loss'], 0.6586655676364899, abs_tol=ABS_TOL)

	def test_bert_base_training_apex_trainer(self):
		""" 
		Tests that we can run a training run with our own trainer.
		We should get the same results as using TrainerBase.
		"""
		print('\ntest_bert_base_training_apex_trainer\n')
		self.set_seed()

		config = Params.from_file('tests/sample_bert_base_config.json')
		config.params['trainer']['accumulation_steps'] = 0
		output_directory = 'tests/bert_base_apex'
		if isdir(output_directory): 
			shutil.rmtree(output_directory)

		trainer = ApexTrainer.from_params(params=config, serialization_dir=output_directory)
		trainer.train()

		# Check that final metrics are correct. This can be useful since different versions 
		# sometimes yield different results and this can potentially reveal this discrepency. 
		final_metrics = json.load(open(join(output_directory, 'metrics_epoch_4.json')))
		assert isclose(final_metrics['training_accuracy'], 0.8418181818181818, abs_tol=ABS_TOL)
		assert isclose(final_metrics['best_validation_accuracy'], 0.8, abs_tol=ABS_TOL)
		assert isclose(final_metrics['training_loss'], 0.722174068291982, abs_tol=ABS_TOL)
		assert isclose(final_metrics['best_validation_loss'], 0.6586655676364899, abs_tol=ABS_TOL)

	def test_bert_base_training_apex_trainer_half_prec_grad_accum(self):
		""" 
		Tests that we can run a training run with our own trainer 
		with half precision training and gradient accumulation.
		"""
		print('\test_bert_base_training_apex_trainer_half_prec_grad_accum\n')
		self.set_seed()

		config = Params.from_file('tests/sample_bert_base_config.json')
		config.params['trainer']['accumulation_steps'] = 2
		config.params['trainer']['half_precision'] = True
		config.params['trainer']['opt_level'] = 'O2'
		output_directory = 'tests/bert_base_apex_half_prec_grad_accum'
		if isdir(output_directory): 
			shutil.rmtree(output_directory)

		trainer = ApexTrainer.from_params(params=config, serialization_dir=output_directory)
		trainer.train()

		# Check that final metrics are correct. This can be useful since different versions 
		# sometimes yield different results and this can potentially reveal this discrepency. 
		final_metrics = json.load(open(join(output_directory, 'metrics_epoch_4.json')))
		assert isclose(final_metrics['training_accuracy'], 0.8418181818181818, abs_tol=ABS_TOL)
		assert isclose(final_metrics['best_validation_accuracy'], 0.9, abs_tol=ABS_TOL)
		assert isclose(final_metrics['training_loss'], 0.7174956496211066, abs_tol=ABS_TOL)
		assert isclose(final_metrics['best_validation_loss'], 0.6359161138534546, abs_tol=ABS_TOL)