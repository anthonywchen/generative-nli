from allennlp.models.archival import load_archive
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.predictors import Predictor
import argparse
from glob import glob
from jsonlines import Reader
from json import dumps, load
import os
from os.path import isdir, join
from statistics import mean, stdev
from tqdm import tqdm

from src.bert import BertNLI
from src.gnli import GNLI
from src.bert_dataset_reader import BertNLIDatasetReader
from src.gnli_dataset_reader import GNLIDatasetReader

# Batch size to do evaluation
BATCH_SIZE = 15

def load_predictor(serialization_dir, device):
	## Load the model
	archive = load_archive(join(serialization_dir, 'model.tar.gz'))
	model = archive.model.eval()
	if device >= 0: 
		model.to(0)

	## Load the dataset reader
	dataset_reader_params = archive.config.pop('dataset_reader')
	model_name = archive.config.pop('model')['type']

	# Turn off truncation of the inputs
	if model_name == 'gnli':
		dataset_reader_params.params['max_premise_length'] = None
		dataset_reader_params.params['max_hypotheis_length'] = None
	elif model_name == 'bertnli':
		dataset_reader_params.params['max_seq_length'] = None
	else:
		raise ValueError()

	reader = DatasetReader.by_name(dataset_reader_params.pop('type')).from_params(dataset_reader_params)

	predictor = Predictor(model, reader)
	return predictor

def is_correct(output_dict, label, dataset):
	correct = False
	# For these datasets, the label is either entails, neutral, or contradicts so 
	# we can directly use the probabilities from the model.
	if dataset in ['anli', 'bizarro', 'mnli']:
		assert label in ['entailment', 'neutral', 'contradiction']
		if (output_dict['predicted_label'] == 0 and label == 'entailment') or \
			(output_dict['predicted_label'] == 1 and label == 'neutral') or \
			(output_dict['predicted_label'] == 2 and label == 'contradiction'):
			correct = True

	# For these datasets, the label is either entails or "not entails" so 
	# we sum the neutral and contradiction probs as the "not entails" probability
	elif dataset in ['hans', 'rte', 'scitail']:
		assert label in ['entailment', 'not_entailment']
		class_probs = output_dict['class_probabilities']
		entail_prob 	= class_probs[0]
		not_entail_prob = class_probs[1] + class_probs[2]

		if (entail_prob >= not_entail_prob and label == 'entailment') or \
		   (entail_prob <  not_entail_prob and label == 'not_entailment'):
			correct = True
	else:
		raise ValueError('Dataset not defined')

	return correct

def predict_file(predictor, file_path, serialization_dir):
	dataset = file_path.split('/')[1]
	print('Scoring ', dataset)

	# Load in all data points first
	instances, tags, labels = [], [], []

	for line in Reader(open(file_path)):
		instances.append(predictor._dataset_reader.text_to_instance(premise=line['premise'], hypothesis=line['hypothesis']))
		labels.append(line['label'])
		if 'tag' in line:
			tags.append(line['tag'])
		else:
			tags.append(None)
	
	total = len(instances)

	# Feed instances through model in batches
	output_dicts = []

	for start_idx in tqdm(range(0, total, BATCH_SIZE)):
		batch_instances = instances[start_idx:start_idx+BATCH_SIZE]
		output_dicts += predictor.predict_batch_instance(batch_instances)
	assert len(output_dicts) == len(labels) == len(tags) == total

	# Compute accuracy statistics
	tag_set = set()
	tag_scores = {}
	num_correct = 0

	for output_dict, tag, label in zip(output_dicts, tags, labels):
		if tag:
			if tag not in tag_set:
				tag_set.add(tag)
				tag_scores[tag+'_correct'] = 0
				tag_scores[tag+'_total'] = 0
			tag_scores[tag+'_total'] += 1

		if is_correct(output_dict, label, dataset):
			if tag: 
				tag_scores[tag+'_correct'] += 1
			num_correct += 1

	results_dict = {'accuracy_'+tag: 100*tag_scores[tag+'_correct']/tag_scores[tag+'_total'] for tag in tag_set}
	results_dict.update(tag_scores)
	results_dict['num_correct'] = num_correct
	results_dict['total'] = total
	results_dict['accuracy'] = 100*num_correct/total

	return results_dict

def predict_run(serialization_dir, device):
	os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

	predictor = load_predictor(serialization_dir, device)

	results_dict = {'anli': 	predict_file(predictor, 'data/anli/test.jsonl', serialization_dir),
					'bizarro': 	predict_file(predictor, 'data/bizarro/test.jsonl', serialization_dir),
					'hans': 	predict_file(predictor, 'data/hans/test.jsonl', serialization_dir),
					'rte': 		predict_file(predictor, 'data/rte/test.jsonl', serialization_dir),
					'scitail': 	predict_file(predictor, 'data/scitail/test.jsonl', serialization_dir),
					'mnli_dev': predict_file(predictor, 'data/mnli/dev.jsonl', serialization_dir)}

	with open(join(serialization_dir, 'generalization_metrics.json'), 'w') as writer:
		writer.write(dumps(results_dict, indent=4, sort_keys=True))

def main(serialization_dir, device):
	num_runs = len(glob(join(serialization_dir, '*') + '/'))

	# Compute generalization metrics for individual runs
	for run_num in range(num_runs):
		print('\n', '='*30, 'Run num', run_num, '='*30)
		predict_run(join(serialization_dir, str(run_num)), device)
		
	# Aggregate generalization metrics across runs
	metrics_dict = {}
	for run_num in range(num_runs):
		cur_metrics_dict = load(open(join(serialization_dir, str(run_num), 'generalization_metrics.json')))

		# Add keys to aggregated generalization metrics dictionary
		if metrics_dict == {}:
			for dataset in cur_metrics_dict:
				metrics_dict[dataset] = {}
				for metric in cur_metrics_dict[dataset]:
					if 'accuracy' in metric:
						metrics_dict[dataset][metric] = []

		# Get scores
		for dataset in metrics_dict:
			for metric in metrics_dict[dataset]:
				metrics_dict[dataset][metric].append(cur_metrics_dict[dataset][metric])

	# Compute average and stdev across runs
	for dataset in metrics_dict:
		for metric in metrics_dict[dataset]:
			mean_metric = str(round(mean(metrics_dict[dataset][metric]), 1))
			stdev_metric = str(round(stdev(metrics_dict[dataset][metric]), 1))
			metrics_dict[dataset][metric] = mean_metric + ' +- ' +  stdev_metric

	# Write aggregated generalization metrics
	with open(join(serialization_dir, 'aggregated_generalization_metrics.json'), 'w') as writer:
		writer.write(dumps(metrics_dict, indent=4, sort_keys=True))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-s', '--serialization_dir', type=str, help='path to serialization_dir with the different runs')
	parser.add_argument('-d', '--device', type=int, default=-1, help='GPU to use. Default is -1 for CPU')
	args = parser.parse_args()

	main(args.serialization_dir, args.device)