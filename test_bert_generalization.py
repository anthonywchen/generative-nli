import argparse
from json import load, loads, dumps
import os
from os.path import isdir, join

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-s', '--serialization_dir', type=str, 
							help='path to serialization_dir with the different runs')
	parser.add_argument('-d', '--device', type=int, default=-1, 
							help='GPU to use. Default is -1 for CPU')
	
	args = parser.parse_args()
	return args

def main():
	args = get_args()

if __name__ == '__main__':
	main()