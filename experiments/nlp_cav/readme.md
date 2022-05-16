# Quantitative Testing with Concept Activation Vectors (TCAV) in NLP

## Data preparation

We use [20-Newsgroup](https://www.kaggle.com/crawford/20-newsgroups) dataset in our experiments.

1. Create `data_path` folder and put `20_newsgroup_complete.csv` file in it
2. Choose labels to experiment with and split the data into train and test by running `lenta_dataset.ipynb`
3. Finally, run `python data_prep.py -dd data_path`. 
This command will save `full.csv`, `train.csv`, `eval.csv`, `embed.pkl` and `vocabs.txt` files to `data_path`

## Training

1. Modify `build_model` function if you want to change an architecture of the model
2. Create `experiment_path` folder and put `experiments/config.yaml` in it
3. Modify hyperparameters inside `experiment_path/config.yaml`
4. Run `python train.py -dd data_path -md experiment_path`. This command will train the model and save checkpoints to `experiment_path`

## Create concepts

1. Choose words you would like to experiments with. For example, `Family`, `Hierarchy`, `Drugs` will be a good choice.
2. Run `python collect_concepts.py -dd data_path -md experiment_path --ngrams 3`. This command will generate multiple files:

* `concepts.pkl` -- for each concept (e.g. `Family`) we search for sentences where this word occurs.
Then we retrieve ngrams of size `n` from this sentence (e.g. `my family does`, `family is the priority`) and call it concepts.
Also we collect some random samples from the data for each concept. 
* `cav_bottlenecks.pkl` -- we convert concept texts into hidden representations of the model from `experiment_path` folder
* `cavs.pkl` -- hyperplanes for each concept received by fitting Logistic Regression on `concept/non-concept` data.
LR is trained on hidden representation of the data. This can be changed to the embedding space.
* `grads.pkl` -- directional derivatives (see the paper for more details)

## Calculate TCAV scores

1. Run `python calculate_tcav.py -dd data_path`. This command will save `scores.pkl` file. 
In this file you can find TCAV scores for each concept against all labels.

## Plot graphs

Run `TCAV.ipynb` to compare results.
