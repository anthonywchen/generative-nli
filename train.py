"""
A wrapper for `allennlp train` command with the ability to run experiments
using different seeds.

The parameters here are passed to a `allennlp train` command after
two steps.

1.	Checking the remote GitHub repository is up-to-date with the local repository

2. 	Creating the serialization directory passed as an argument

3. 	Iterating through the number of different seeds to try (from `num_runs` argument), 
	modifying the seed after each run. 

	For each run, we create and run an `allennlp train`
	command by stitching together the arguments passed by the user in `get_train_args()`.

	For example, if `serialization_dir` is `./tmp` and `num_runs` is 2, we will create two 
	subdirectories, `./tmp/0` and `./tmp/1` and store two training runs in the two subdirectories
	with different random seeds.
"""
from allennlp.common.params import parse_overrides
from allennlp.commands.train import train_model_from_file
from allennlp.common.util import import_submodules
import argparse
import git
from json import load, loads, dumps
from _jsonnet import evaluate_file
import logging
import os
from os.path import isdir, isfile, join
from statistics import mean, stdev
import sys

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

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

def get_train_args():
	parser = argparse.ArgumentParser()
	description = '''Train the specified model on the specified dataset.'''
	parser.add_argument('param_path', type=str,
						   help='path to parameter file describing the model to be trained')
	parser.add_argument('--include-package', type=str, action='append', default=[],
						   help='additional packages to include')
	parser.add_argument('-s', '--serialization-dir', required=True, type=str,
						   help='directory in which to save the model and its logs')
	parser.add_argument('-o', '--overrides', type=str, default="", 
						   help='a JSON structure used to override the experiment configuration')
	parser.add_argument('--num_runs', type=int, required=True,
						   help='Number of times to run the experiment (using different seeds)')
	parser.add_argument('--sha', type=str,
						   help='Hash of the current commit')

	args = parser.parse_args()
	args_dict = vars(args)
	return args_dict

def aggregate_training_run_metrics(head_serialization_dir, num_runs):
	""" Aggregates validation metrics across the training runs """
	metrics_dict = {key: [] for key in ['best_validation_accuracy', 'best_validation_loss']}

	# Gets validation metrics across runs
	for run_number in range(num_runs):
		cur_metrics_dict = load(open(join(head_serialization_dir, str(run_number), 'metrics.json')))
		for key in metrics_dict:
			metrics_dict[key].append(cur_metrics_dict[key])

	# Computes mean and std deviation of scores across runs
	for key in metrics_dict:
		scores = metrics_dict[key]
		metrics_dict[key] = str(round(100*mean(scores), 1)) + ' +- ' +  str(round(100*stdev(scores), 1))

	output_file = join(head_serialization_dir, 'aggregated_metrics.json')
	with open(output_file, 'w') as writer:
		writer.write(dumps(metrics_dict, indent=4, sort_keys=True))

def setup_environment(config, include_package, finetune_source, serialization_dir, sha):
	# Get the commit hash if it hasn't been passed in
	sha = sha if sha else get_commit_hash()
	
	# Set the cuda device as a environment variable, since there are issues
	# setting the GPU as GPU0 with apex.
	if 'cuda_device' in config['trainer']:
		os.environ["CUDA_VISIBLE_DEVICES"] = str(config['trainer']['cuda_device'])
		config['trainer']['cuda_device'] = 0

	# Create the (head) serialization directory
	head_serialization_dir = serialization_dir
	if not isdir(serialization_dir):
		os.mkdir(serialization_dir)

	# Write out the current commit hash to file
	with open(join(serialization_dir, 'hash.txt'), 'w') as f:
		f.write(sha)

	# Write out the current commit hash to file
	if finetune_source:
		with open(join(serialization_dir, 'finetune_source.txt'), 'w') as f:
			f.write(finetune_source + '\n')

	# Import any additional modules needed (to register custom classes).
	for package_name in include_package:
		import_submodules(package_name)

def train(param_path, serialization_dir, num_runs, overrides = "", include_package = [], sha = "", finetune_source=""):
	# Load config file if it isn't passed in
	config = parse_overrides(overrides) if overrides else loads(evaluate_file(param_path))
	
	# Check that we have `num_runs` pretrained models under `finetune_source`
	if finetune_source:
		for run_number in range(int(num_runs)):
			assert isfile(join(finetune_source, str(run_number), 'model.tar.gz'))

	setup_environment(config, include_package, finetune_source, serialization_dir, sha)

	# Iterate through the runs, modifying the seeds per run
	head_serialization_dir = serialization_dir
	for run_number in range(int(num_runs)):
		# Write outputs to `head_serialization_dir/run_number`
		serialization_dir = join(head_serialization_dir, str(run_number))
		
		if finetune_source:
			config['trainer']['pretrained_model'] = join(finetune_source, str(run_number), 'model.tar.gz')

		train_model_from_file(parameter_filename=param_path, 
							  serialization_dir=serialization_dir,
							  overrides=dumps(config))

		# Grab the seeds and write them to file, since allennlp doesn't save seeds in `serialization_dir`
		with open(join(serialization_dir, 'seeds.json'), 'w') as f:
			f.write(dumps({k: config[k] for k in ('pytorch_seed','random_seed','numpy_seed')}) + '\n')

		# Add 10 to the seeds in the config for the next run
		config['numpy_seed'] += 10
		config['random_seed'] += 10
		config['pytorch_seed'] += 10

	aggregate_training_run_metrics(head_serialization_dir, num_runs)

if __name__ == '__main__':
	args = get_train_args()
	train(**args)