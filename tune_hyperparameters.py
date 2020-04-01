import argparse
import collections
import git
from json import load, loads, dumps
from _jsonnet import evaluate_file
import itertools
import logging
from os.path import isdir, join
import sys

from train import train

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

TUNABLE_PARAMETERS = ['batch_size', 'disc_loss_weight', 'lr', 'num_epochs', 'weight_decay', 'warmup_proportion']

class ConfigIterator:
	def __init__(self, config):
		self.flat_config = self.flatten(config)
		self.hp = self.search() # hyperparaters dict
		self.hp_keys = self.hp.keys()
		self.hp_combinations = list(itertools.product(*self.hp.values()))
		self.n = len(self.hp_combinations)
		self.current = 0

	def flatten(self, d, parent_key=''):
		items = []

		for k, v in d.items():
			new_key = parent_key + '.' + k if parent_key else k

			if isinstance(v, collections.MutableMapping):
				items.extend(self.flatten(v, new_key).items())
			else:
				items.append((new_key, v))

		return dict(items)

	def unflatten(self, dictionary):
		resultDict = {}
		for key, value in dictionary.items():
			parts = key.split('.')
			d = resultDict
			for part in parts[:-1]:
				if part not in d:
					d[part] = dict()

				d = d[part]

			d[parts[-1]] = value

		return resultDict

	def search(self):
		hp = {} # hyperparameters dictionary
		for k, v in self.flat_config.items():
			if type(v) == list and k.split('.')[-1] in TUNABLE_PARAMETERS:
				hp[k] = v
		return hp

	def __iter__(self):
		return self

	def __next__(self):
		if self.current >= self.n:
			raise StopIteration
		else:
			self.current += 1
			flat_config = self.flat_config
			params = dict(zip(self.hp_keys, self.hp_combinations[self.current-1]))

			for k, v in params.items():
				flat_config[k] = v

			config = self.unflatten(flat_config)
			return config, params
			
def get_commit_hash():
	"""
	Check that we have pushed all changes from the local repo to GitHub
	so that we can reference the commit hash in the trained directory.
	
	Returns: 
	sha: `string` the current hash of HEAD commite
	"""
	g = git.cmd.Git('.')
	for status_line in g.status().split('\n'):
		if 'modified:' in status_line:
			if 'configs/' not in status_line:
				print('Changes not yet committed')
				sys.exit(1)
			else:
				print(status_line)

	# Get current hash of HEAD
	repo = git.Repo(search_parent_directories=True)
	sha = repo.head.object.hexsha

	return sha

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('param_path', type=str, help='path to parameter file describing the model to be trained')
	parser.add_argument('-s', '--serialization_dir', type=str, help='path to serialization_dir with the different runs')
	parser.add_argument('-n', '--num_runs', type=int, default=3, help='# of different runs')
	parser.add_argument('-f', '--finetune_source', type=str, default="", help='path to directory that we can finetune')
	args = parser.parse_args()

	sha = get_commit_hash()
	config = loads(evaluate_file(args.param_path))
	config_iterator = ConfigIterator(config)

	for c, params in config_iterator:
		# Name this config's serialization directory and check that it doesn't exist

		serialization_dir = '_'.join([str(v) + '_' + k.split('.')[-1] for k, v in params.items()])
		if args.finetune_source:
			serialization_dir = 'finetune_' + serialization_dir
		serialization_dir = join(args.serialization_dir, serialization_dir)

		train(param_path=args.param_path,
			  serialization_dir=serialization_dir,
			  num_runs=args.num_runs,
			  overrides=dumps(c),
			  include_package=['src'],
			  sha=str(sha),
			  finetune_source=args.finetune_source)

if __name__ == '__main__':
	main()