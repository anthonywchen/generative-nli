import csv
from jsonlines import Reader
import json
import os
from os.path import dirname, isdir, join

def create_dir(directory_name):
	# Create data directory if it doesn't exist
	if not isdir(directory_name):
		os.mkdir(directory_name)

def process_snli_mnli(input_file, output_file):
	create_dir(dirname(output_file))

	with open(output_file, 'w') as writer:
		for line in Reader(open(input_file)):
			label = line['gold_label']
			if label not in ['entailment', 'neutral', 'contradiction']:
				continue
			
			output_line = {'premise': line['sentence1'],
						   'hypothesis': line['sentence2'],
						   'label': label}
			writer.write(json.dumps(output_line) + '\n')

def process_rte(input_dir, output_file):
	create_dir(dirname(output_file))

	def process_file(input_file, tag):
		processed_lines = []

		for line in Reader(open(input_file)):
			label = line['label']
			assert label in ['entailment', 'not_entailment']

			output_line = {'premise': line['premise'], 'hypothesis': line['hypothesis'],
						   'label': label, 'tag': tag}
			processed_lines.append(output_line)

		return processed_lines

	processed_lines = process_file(join(input_dir, 'train.jsonl'), tag='train')
	processed_lines += process_file(join(input_dir, 'val.jsonl'), tag='dev')

	with open(output_file, 'w') as writer:
		for line in processed_lines:
			writer.write(json.dumps(line) + '\n')

def process_scitail(input_file, output_file):
	create_dir(dirname(output_file))

	with open(output_file, 'w') as writer:
		with open(input_file) as f:
			for line in f:
				premise, hypothesis, label = line.strip().split('\t')
				if label == 'entails':
					label = 'entailment'
				elif label == 'neutral':
					label = 'not_entailment'
				else:
					raise ValueError()

				output_line = {'premise': premise, 
							   'hypothesis': hypothesis,
							   'label': label,
							   'tag': ''}
				writer.write(json.dumps(output_line) + '\n')

def process_anli(input_dir, output_file):
	create_dir(dirname(output_file))

	def process_file(input_file):
		processed_lines = []
		for line in Reader(open(input_file)):
			label = line['label']
			assert label in ['e', 'n', 'c']
			if label == 'e':
				label = 'entailment'
			elif label == 'n':
				label = 'neutral'
			else:
				label = 'contradiction'

			output_line = {'premise': line['context'],
						   'hypothesis': line['hypothesis'],
						   'label': label,
						   'tag': line['tag']}
			processed_lines.append(output_line)

		return processed_lines

	processed_lines = process_file(join(input_dir, 'R1/test.jsonl'))
	processed_lines += process_file(join(input_dir, 'R2/test.jsonl'))
	processed_lines += process_file(join(input_dir, 'R3/test.jsonl'))

	with open(output_file, 'w') as writer:
		for line in processed_lines:
			writer.write(json.dumps(line) + '\n')

def process_bizarro(input_dir, output_file):
	create_dir(dirname(output_file))

	def process_file(input_file):
		processed_lines = []

		for i, line in enumerate(open(input_file)):
			if i == 0: # Skip header line
				continue
				
			premise, hypothesis, label = line.strip().split('\t')
			assert label in ['entailment', 'neutral', 'contradiction']

			output_line = {'premise': premise,
						   'hypothesis': hypothesis,
						   'label': label,
						   'tag': dirname(input_file)}
			processed_lines.append(output_line)

		return processed_lines

	processed_lines = process_file(join(input_dir, 'revised_premise/test.tsv'))
	processed_lines += process_file(join(input_dir, 'revised_hypothesis/test.tsv'))

	with open(output_file, 'w') as writer:
		for line in processed_lines:
			writer.write(json.dumps(line) + '\n')

def process_hans(input_file, output_file):
	create_dir(dirname(output_file))

	with open(output_file, 'w') as writer:

		with open(input_file, newline='') as f:
			f.readline() # Skip header
			for line in csv.reader(f, delimiter='\t'):
				label = line[0]
				assert label in ['entailment', 'non-entailment']

				if label == 'non-entailment':
					label = 'not_entailment'

				output_line = {'premise': line[5],
			   				   'hypothesis': line[6],
			   				   'label': label,
			   				   'tag': line[8]+'_'+label}

				writer.write(json.dumps(output_line) + '\n')

def main():
	print('PROCESSING SNLI...')
	process_snli_mnli('raw_data/snli/snli_1.0_train.jsonl', 'data/snli/train.jsonl')
	process_snli_mnli('raw_data/snli/snli_1.0_dev.jsonl', 'data/snli/dev.jsonl')

	print('PROCESSING MNLI...')
	process_snli_mnli('raw_data/mnli/multinli_1.0_train.jsonl', 'data/mnli/train.jsonl')
	process_snli_mnli('raw_data/mnli/multinli_1.0_dev_matched.jsonl', 'data/mnli/dev.jsonl')
	
	print('PROCESSING RTE...')
	process_rte('raw_data/rte', 'data/rte/test.jsonl')

	print('PROCESSING SCITAIL...')
	process_scitail('raw_data/scitail/tsv_format/scitail_1.0_test.tsv', 'data/scitail/test.jsonl')

	print('PROCESSING ANLI...')
	process_anli('raw_data/anli', 'data/anli/test.jsonl')

	print('PROCESSING BIZARRO...')
	process_bizarro('raw_data/bizarro', 'data/bizarro/test.jsonl')

	print('PROCESSING HANS...')
	process_hans('raw_data/hans/heuristics_evaluation_set.txt', 'data/hans/test.jsonl')

if __name__ == '__main__':
	main()