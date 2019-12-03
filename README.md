# Class Conditional Generative Modeling of Natural Langauge Inference

To run tests on of the BERT (and ROBERTA) dataset reader and training, run:
```
python -m pytest tests/test_bert_dataset_reader.py -s
python -m pytest tests/test_bert_training.py -s
```

To train a model 
```
python train.py [config_file] --include-package src -s [output_directory] --num_runs [num_runs]
```

## To Do:

- [x] Add modeling for BERT NLI
- [x] Add config for BERT NLI
- [ ] Add training script so that we can do multple seeds
- [ ] Add custom trainer so that we have gradient accumulation and half precision training