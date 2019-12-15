# Class Conditional Generative Modeling of Natural Langauge Inference

## Training and Evaluation of BERT and ROBERTa
To run tests on of the BERT (and ROBERTA) dataset reader and training, run:
```
python -m pytest tests/test_bert_dataset_reader.py -s
python -m pytest tests/test_bert_training.py -s
```

I have a training script that acts as a wrapper around `allennlp train`. 
The two main functionalities it adds is it makes sure all changes are commmitted before running `allennlp train`, 
and also allows you to specify a `--num_runs` parameter which specifies the number of training runs with
different seeds. To use this script, run:
```
python train.py [config_file] --include-package src -s [output_directory] --num_runs 3
```

To evaluate the generalization of a trained model (across runs), run:
```
python test_bert_generalization.py -s [serializatino_dir] -d [device]
```

## Training and Evaluation of BART

## To Do:

- [x] Add modeling for BERT NLI
- [x] Add config for BERT NLI
- [x] Add training script so that we can do multple seeds
- [x] Add custom trainer so that we have gradient accumulation and half precision training
- [x] Aggregation script for different runs
- [x] Evaluation script for evaluating on different datasets
- [ ] Process BIZARRO dataset
- [ ] Add new datasets to test_bert_generalization.py

## Training models:

##### SNLI RESULTS:
- [x] BERT-BASE 	0.897+-0.001
- [x] ROBERTA-BASE 	0.907+-0.001
- [ ] BERT-LARGE
	- [ ] 32 bs 1e-5lr
	- [ ] 32 bs 1e-6lr
- [ ] ROBERTA-LARGE	
	- [ ] 32 bs 1e-5lr
	- [ ] 32 bs 1e-6lr

##### MNLI RESULTS:
- [x] BERT-BASE 	0.827+-0.001
- [x] ROBERTA-BASE 	0.862+-0.003
- [ ] BERT-LARGE
- [x] ROBERTA-LARGE