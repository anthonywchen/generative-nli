# Class Conditional Generative Modeling of Natural Langauge Inference

## Seting up the repository

First run `./setup.sh` in the root of the repository. This creates a virtual environment called `generative-nli-env` and then installs the packages in `requirements.txt`. I wanted pytorch-transformers-1.2.0 but allennlp-0.9.0 supports version 1.1.0 by default. This throws an error but I think it hasn't broken anything.

It also installs APEX for half-precision training. It then downloads all the NLI datasets into `raw_data/`. 

One thing you may have to change are lines 10-11. I have CUDA-10.0 and so I needed that specific PyTorch library. Change this if you have a different version of CUDA. Also I'm using PyTorch-1.2.0 but there may be a newer version.

## Processing the data
The next step is to process the NLI datasets in `raw_data/`. All the NLI datasets have their own format but I wanted them all to be processed into a unified JSONLines file format. To process the data, run: `python data/process_raw_data.py` which will process the datasets in `raw_data/` and write the outputs to `data/`.

## Training of BERT and ROBERTa on MNLI
The configuration files are in `configs/`. 

I have a training script (`train.py`) that acts as a wrapper around `allennlp train`. 

It makes sure all changes are commmitted before running `allennlp train` and writes the hash of the commit into `<serialization_dir>/hash.txt`. If you want to disable this check add the flag `--sha` when calling `train.py` and pass a random string. 

This script also allows you to specify a `--num_runs` parameter which specifies the number of training runs with different seeds (the seeds are incremented by 10 after each training run). To use train a config with three runs, run:
```
python train.py configs/[config_file] --include-package src -s [serialization_dir] --num_runs 3
```
What happens is a directory is created under `serialization_dir` for each run number. I.e. for the first training run, the outputs are stored in `<serialization_dir>/0/`, for the second `<serialization_dir>/1/` and so on.

After completing training for all runs, `train.py` aggregates the validation results on MNLI across the runs into `<serialization_dir>/aggregated_metrics.json`. The results here are the average and standard deviation across the runs. 

It should be noted that the trainer I use in the configs is `src/apex_trainer.py`  which adds gradient accumulation and half precision training on top of the `TrainerBase()` class. If you want to do normal training w/o grad. accum or half-prec training, just comment out `accumulation_steps`, `half_precision`, and `opt_level` under `trainer` in the configs. 

I should also note that I didn't use the official GLUE evaluation script so its possible (?) that I could have a bug in computing these dev scores. But I don't think I messed this up....

Here are the results I got using the configs in `configs/` across three runs. 

| Model | My Results on MNLI Matched Dev | Official Results on MNLI Matched Dev |
| ------------- |:-------------:| :-----:|
| bert-base      | 84.6 ± 0.2 | 84.4 |
| bert-large      | 86.5 ± 0.2      |  86.6|
| roberta-base | 87.7 ± 0.1      |    87.6 |
| roberta-large | 90.6 ± 0.2      |    90.2 |

## Evaluation of BERT and ROBERTa on Out-Of-Domain Datasets
To evaluate the generalization of a trained model (across runs), on out-of-domain datasets, run
```
python test_bert_generalization.py -s [serialization_dir] -d [device]
```
For each run, this evaluates the trained model on out-of-domain datasets (RTE, SciTail, Hans, Bizarro, and ANLI), and aggregates the results across the runs into `<serialization_dir>/aggregated_generalization_metrics.json`. 

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
- [x] Add model for BART
- [x] Are the weights of the encoder and the decoder tied? yes
- [x] How to add 3 new embeddings for the labels?
- [x] Add tests for BART dataset reader
- [x]  Add tests for the calculation of the logits
- [x] Add tests for model training
- [x] Add calculation of losses and the mixing
- [x] More information when calculating generalization
- [x] Why high disc loss but low accuracy - 1 day
- [x] Best LR for bs = 16? - 1 day 
- [x] Train on all data? - 1 day
- [ ] Adding label embed in decoding layer? - 2 day