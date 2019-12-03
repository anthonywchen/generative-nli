# Class Conditional Generative Modeling of Natural Langauge Inference

To run tests on of the BERT (and ROBERTA) dataset reader and training, run:
```
python -m pytest tests/test_bert_dataset_reader.py -s
python -m pytest tests/test_bert_training.py -s
```

To train a model 
```
python train.py [config_file] --include-package src -s [output_directory] --num_runs 3
```

## To Do:

- [x] Add modeling for BERT NLI
- [x] Add config for BERT NLI
- [x] Add training script so that we can do multple seeds
- [s] Add custom trainer so that we have gradient accumulation and half precision training
- [ ] Aggregation script for different runs
- [ ] Evaluation script for evaluating on different datasets

## Training models:
- [x] bert base on SNLI
- [ ] bert base on MNLI
- [ ] roberta base on SNLI
- [ ] roberta base on MNLI

- [ ] bert large on SNLI
- [ ] bert large on MNLI
- [ ] roberta large on SNLI
- [ ] roberta large on MNLI
