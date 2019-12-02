# Class Conditional Generative Modeling of Natural Langauge Inference

To run tests on of the BERT (and ROBERTA) dataset reader and training, run:
```
python -m pytest tests/test_dataset_reader.py -s
python -m pytest tests/test_bert_training.py -s

```

## To Do:

- [ ] Add modeling for BERT NLI
- [ ] Add config for BERT NLI
- [ ] Add training script so that we can do multple seeds
- [ ] Add custom trainer so that we have gradient accumulation and half precision training