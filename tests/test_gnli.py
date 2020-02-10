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
from os.path import isdir, join
import random
import shutil
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
		gnli = GNLI('bart.large')
		bart = torch.hub.load('pytorch/fairseq', 'bart.large').model

		# Check that GNLI encoder and decoder are tied
		assert gnli._bart.encoder.embed_tokens == gnli._bart.decoder.embed_tokens

		# Check that GNLI and original BART token embeddings values match
		assert torch.all(list(gnli._bart.encoder.embed_tokens.embed_tokens.parameters())[0] == list(bart.encoder.embed_tokens.parameters())[0]).item()

	def test_training(self):
		""" 
		Tests that we can run a training run, and 
		record the output scores so that we can always check back 
		"""
		print('\n test_training \n')
		self.set_seed()

		config = Params.from_file('tests/sample_gnli_config.json')
		output_directory = 'tests/gnli'
		if isdir(output_directory): 
			shutil.rmtree(output_directory)

		trainer = ApexTrainer.from_params(params=config, serialization_dir=output_directory)
		trainer.train()

		# Check that final metrics are correct. This can be useful since different versions 
		# sometimes yield different results and this can potentially reveal this discrepency. 
		# final_metrics = json.load(open(join(output_directory, 'metrics_epoch_19.json')))
		# assert isclose(final_metrics['training_accuracy'], 0.8763636363636363, abs_tol=ABS_TOL)
		# assert isclose(final_metrics['best_validation_accuracy'], 0.9, abs_tol=ABS_TOL)
		# assert isclose(final_metrics['best_validation_loss'], 1.19368314743042, abs_tol=ABS_TOL)
		# assert isclose(final_metrics['best_epoch'], 17)