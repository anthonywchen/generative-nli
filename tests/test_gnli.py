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
os.environ["CUDA_VISIBLE_DEVICES"]="1"		
vocab = Vocabulary()

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class Tests(AllenNlpTestCase):
	def set_seed(self):
		random.seed(0)
		torch.manual_seed(0)

	# def test_gnli_training(self):
	# 	""" 
	# 	Tests that we can run a training run, and 
	# 	record the output scores so that we can always check back 
	# 	"""
	# 	print('\ntest_training\n')
	# 	self.set_seed()

	# 	config = Params.from_file('tests/sample_gnli_config.json')
	# 	config.params['trainer']['accumulation_steps'] = 2
	# 	output_directory = 'tests/gnli'
	# 	if isdir(output_directory): 
	# 		shutil.rmtree(output_directory)

	# 	trainer = ApexTrainer.from_params(params=config, serialization_dir=output_directory)
	# 	trainer.train()

	def test_gnli_training_with_disc_loss(self):
		""" 
		Tests that we can run a training run, and 
		record the output scores so that we can always check back 
		"""
		print('\ntest_gnli_training_with_disc_loss\n')
		self.set_seed()

		config = Params.from_file('tests/sample_gnli_config.json')
		config.params['trainer']['accumulation_steps'] = 2
		config.params['model']['discriminative_loss_weight'] = .5
		output_directory = 'tests/gnli_with_disc'
		if isdir(output_directory): 
			shutil.rmtree(output_directory)

		trainer = ApexTrainer.from_params(params=config, serialization_dir=output_directory)
		trainer.train()

	def test_half_prec_grad_accum(self):
		""" 
		Tests that we can run a training run with our own trainer 
		with half precision training and gradient accumulation.
		"""
		pass
