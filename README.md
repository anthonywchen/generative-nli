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
python test_bert_generalization.py -s [serialization_dir] -d [device]
```

## Training and Evaluation of BART

## To Do:

- [x] Add modeling for BERT NLI
- [x] Add config for BERT NLI
- [x] Add training script so that we can do multple seeds
- [x] Add custom trainer so that we have gradient accumulation and half precision training
- [x] Aggregation script for different runs
- [x] Evaluation script for evaluating on different datasets
- [x] Process BIZARRO dataset
- [x] Add new datasets to test_bert_generalization.py
- [x] Add dataset reader for BART
- [ ] Add model for BART
- [x] Are the weights of the encoder and the decoder tied? yes
- [ ] How to add 3 new embeddings for the labels?
- [x] Add tests for BART dataset reader


