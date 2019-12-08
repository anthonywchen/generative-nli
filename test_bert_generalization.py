from allennlp.models.archival import load_archive
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.predictors import Predictor
import argparse
from glob import glob
from jsonlines import Reader
from json import dumps, load
from os.path import isdir, join
from pprint import pprint
from statistics import mean, stdev
from tqdm import tqdm

from src.bert_dataset_reader import BertNLIDatasetReader
from src.bert import BertNLI

def load_predictor(serialization_dir, device):
	archive = load_archive(join(serialization_dir, 'model.tar.gz'))
	model = archive.model.eval()
	if device >= 0: 
		model.to(device)

	dataset_reader_params = archive.config.pop('dataset_reader')
	reader = DatasetReader.by_name(dataset_reader_params.pop('type')).from_params(dataset_reader_params)
	
	return Predictor(model, reader)

def is_correct(output_dict, label, dataset):
	correct = False
	if dataset in ['anli']:
		if (output_dict['predicted_label'] == 0 and label == 'entailment') or \
			(output_dict['predicted_label'] == 1 and label == 'neutral') or \
			(output_dict['predicted_label'] == 2 and label == 'contradiction'):
			correct = True
	
	elif dataset in ['hans', 'rte']:
		class_probs = output_dict['class_probabilities']
		entail_prob, not_entail_prob = class_probs[0], class_probs[1]+class_probs[2]
		if (entail_prob >= not_entail_prob and label == 'entailment') or \
		   (entail_prob < not_entail_prob and label == 'not_entailment'):
			correct=True
	else:
		raise ValueError('Dataset not defined')

	return correct

def predict_file(predictor, file_path, serialization_dir):
	dataset = file_path.split('/')[1]
	print('Scoring ', dataset)

	tags = set()
	tag_scores = {}
	num_correct = 0
	total = 0

	for line in tqdm(Reader(open(file_path))):
		instance = predictor._dataset_reader.text_to_instance(premise=line['premise'], hypothesis=line['hypothesis'])
		output_dict = predictor.predict_instance(instance)
		total += 1

		tag = line['tag']
		if tag:
			if tag not in tags:
				tags.add(tag)
				tag_scores[tag+'_correct'] = 0
				tag_scores[tag+'_total'] = 0
			tag_scores[tag+'_total'] += 1

		if is_correct(output_dict, line['label'], dataset):
			if tag: 
				tag_scores[tag+'_correct'] += 1
			num_correct += 1
	
	results_dict = {'accuracy_'+tag: tag_scores[tag+'_correct']/tag_scores[tag+'_total'] for tag in tags}
	results_dict.update(tag_scores)
	results_dict['num_correct'] = num_correct
	results_dict['total'] = total
	results_dict['accuracy'] = num_correct/total

	return results_dict

def predict_run(serialization_dir, device):
	predictor = load_predictor(serialization_dir, device)
	results_dict = {}

	results_dict['anli'] = predict_file(predictor, 'data/anli/test.jsonl', serialization_dir)
	results_dict['hans'] = predict_file(predictor, 'data/hans/test.jsonl', serialization_dir)
	results_dict['rte'] = predict_file(predictor, 'data/rte/test.jsonl', serialization_dir)

	with open(join(serialization_dir, 'generalization_metrics.json'), 'w') as writer:
		writer.write(dumps(results_dict, indent=4, sort_keys=True))

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-s', '--serialization_dir', type=str, help='path to serialization_dir with the different runs')
	parser.add_argument('-d', '--device', type=int, default=-1, help='GPU to use. Default is -1 for CPU')
	args = parser.parse_args()

	num_runs = len(glob(join(args.serialization_dir, '*') + '/'))

	# Compute generalization metrics for individual runs
	for run_num in range(num_runs):
		print('\n', '='*30, 'Run num', run_num, '='*30)
		predict_run(join(args.serialization_dir, str(run_num)), args.device)
		
	# Aggregate generalization metrics across runs
	metrics_dict = {}
	for run_num in range(num_runs):
		cur_metrics_dict = load(open(join(args.serialization_dir, str(run_num), 'generalization_metrics.json')))

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
			mean_metric = str(round(mean(metrics_dict[dataset][metric]), 3))
			stdev_metric = str(round(stdev(metrics_dict[dataset][metric]), 3))
			metrics_dict[dataset][metric] = mean_metric + '+-' +  stdev_metric

	# Write aggregated generalization metrics
	with open(join(args.serialization_dir, 'aggregated_generalization_metrics.json'), 'w') as writer:
		writer.write(dumps(metrics_dict, indent=4, sort_keys=True))

if __name__ == '__main__':
	main()