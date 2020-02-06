""" Functions for reading and sampling data. 
	I have two dataset readers and I want the reading/sampling to be consistent across the two
"""
from jsonlines import Reader
import math
import logging

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def read_data(file_path, percent_data):
	# Load in all lines
	with open(file_path) as f:
		lines = [line for line in Reader(f)]

	# Determine how many lines we will use as a percent of the data
	num_lines_to_use = math.ceil(len(lines)*percent_data)
	logger.info('Number of data points: %d', num_lines_to_use)
	
	if percent_data < 1:
		lines = sample_data(lines, num_lines_to_use)

	# Print out the class balance
	counts = {'entailment': 0, 'neutral': 0, 'contradiction': 0}
	for l in lines:
		counts[l['label']] += 1
	logger.info('Counts by label: ' + counts.__repr__())

	return lines

def sample_data(lines, num_lines_to_use):
	""" Sample data, while ensuring class balance """
	logger.info('Sampling lines...')
	
	# Group lines by their label
	lines_by_label = {'entailment': [], 'neutral': [], 'contradiction': []}
	for line in lines:
		lines_by_label[line['label']].append(line)

	# Sample data, adding one data point per label at a time
	new_lines = []
	for i in range(num_lines_to_use):
		if i % 3 == 0:
			new_lines.append(lines_by_label['entailment'].pop())
		elif i % 3 == 1:
			new_lines.append(lines_by_label['neutral'].pop())
		else:
			assert i % 3 == 2
			new_lines.append(lines_by_label['contradiction'].pop())
	
	return new_lines