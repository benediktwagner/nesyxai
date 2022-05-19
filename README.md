# Accompanying Repository to the PhD thesis: Reasoning about what has been learned
## Neural-Symbolic Integration for explainable Artificial Intelligence

The thesis builds upon the Logic Tensor Network (LTN) approach. It is a neurosymbolic framework that supports querying, learning and reasoning with both rich data and rich abstract knowledge about the world.
LTN uses a differentiable first-order logic language, called Real Logic, to incorporate data and logic. 
This PyTorch adaptation follows the main [Tensorflow implementation](https://github.com/logictensornetworks/logictensornetworks). 

The repository consists of pracitcal illustration and tutorials to utilise the LTN framework in an interactive manner. The tutorials follow the ones developed for the official Tensorflow LTN library. This implementation, however, is fully translated into the PyTorch framework with additional functionality (such as simple integration of different types of existing models and linear probing of layers for grounding abstract representations). 
All experiments are added or will be added once they are cleaned and compatible with the larger code-base.

![Grounding_illustration](./docs/img/framework_grounding.png)

LTN converts Real Logic formulas (e.g. `∀x(cat(x) → ∃y(partOf(x,y)∧tail(y)))`) into [PyTorch](https://www.pytorch.org/) computational graphs.
Such formulas can express complex queries about the data, prior knowledge to satisfy during learning, statements to prove ...

![Computational_graph_illustration](./docs/img/framework_computational_graph.png)

One can represent and effectively compute the most important tasks of deep learning. Examples of such tasks are classification, regression, clustering, or link prediction.
The ["Getting Started"](#getting-started) section of the README links to tutorials and examples of LTN code.

Extensive [[Paper]](https://arxiv.org/pdf/2012.13635.pdf) covering the theory behind this code. 

## Installation

Clone the LTN repository and install it using `pip install -e <local project path>`.

Following are the dependencies we used for development (similar versions should run fine):
- python 3.8
- pytorch >= 1.0 (for running the core system)
- numpy >= 1.18 (for examples)
- matplotlib >= 3.2 (for examples)

## Repository structure

- `logictensornetworks/core.py` -- core system for defining constants, variables, predicates, functions and formulas,
- `logictensornetworks/fuzzy_ops.py` -- a collection of fuzzy logic operators defined using Tensorflow primitives,
- `logictensornetworks/utils.py` -- a collection of useful functions,
- `tutorials/` -- tutorials to start with LTN,
- `examples/` -- various problems approached using LTN,
- `tests/` -- tests.

## Getting Started

### Tutorials

`tutorials/` contains a walk-through of LTN. In order, the tutorials cover the following topics:
1. [Grounding in LTN part 1](https://nbviewer.jupyter.org/github/logictensornetworks/logictensornetworks/blob/master/tutorials/1-grounding_non_logical_symbols.ipynb): Real Logic, constants, predicates, functions, variables,
2. [Grounding in LTN part 2](https://nbviewer.jupyter.org/github/logictensornetworks/logictensornetworks/blob/master/tutorials/2-grounding_connectives.ipynb): connectives and quantifiers (+ [complement](https://nbviewer.jupyter.org/github/logictensornetworks/logictensornetworks/blob/master/tutorials/2b-operators_and_gradients.ipynb): choosing appropriate operators for learning),
3. [Learning in LTN](https://nbviewer.jupyter.org/github/logictensornetworks/logictensornetworks/blob/master/tutorials/3-knowledgebase_and_learning.ipynb): using satisfiability of LTN formulas as a training objective,
4. [Reasoning in LTN](https://nbviewer.jupyter.org/github/logictensornetworks/logictensornetworks/blob/master/tutorials/4-reasoning.ipynb): measuring if a formula is the logical consequence of a knowledgebase.

The tutorials are implemented using jupyter notebooks.

### Examples

`examples/` contains a series of experiments. Their objective is to show how the language of Real Logic can be used to specify a number of tasks that involve learning from data and reasoning about logical knowledge. Examples of such tasks are: classification, regression, clustering, link prediction.

- The [binary classification](https://nbviewer.jupyter.org/github/logictensornetworks/logictensornetworks/blob/master/examples/binary_classification/binary_classification.ipynb) example illustrates in the simplest setting how to ground a binary classifier as a predicate in LTN, and how to feed batches of data during training,
- The multiclass classification examples ([single-label](https://nbviewer.jupyter.org/github/logictensornetworks/logictensornetworks/blob/master/examples/multiclass_classification/multiclass-singlelabel.ipynb), [multi-label](https://nbviewer.jupyter.org/github/logictensornetworks/logictensornetworks/blob/master/examples/multiclass_classification/multiclass-multilabel.ipynb)) illustrate how to ground predicates that can classify samples in several classes,
- The [MNIST digit addition](https://nbviewer.jupyter.org/github/logictensornetworks/logictensornetworks/blob/master/examples/mnist/single_digits_addition.ipynb) example showcases the power of a neurosymbolic approach in a classification task that only provides groundtruth for some final labels (result of the addition), where LTN is used to provide prior knowledge about intermediate labels (possible digits used in the addition),
- The [regression](https://nbviewer.jupyter.org/github/logictensornetworks/logictensornetworks/blob/master/examples/regression/regression.ipynb) example illustrates how to ground a regressor as a function symbol in LTN,
- The [clustering](https://nbviewer.jupyter.org/github/logictensornetworks/logictensornetworks/blob/master/examples/clustering/clustering.ipynb) example illustrates how LTN can solve a task using first-order constraints only, without any label being given through supervision,
- The [Smokes Friends Cancer](https://nbviewer.jupyter.org/github/logictensornetworks/logictensornetworks/blob/master/examples/smokes_friends_cancer/smokes_friends_cancer.ipynb) example is a classical link prediction problem of Statistical Relational Learning where LTN learns embeddings for individuals based on fuzzy groundtruths and first-order constraints.

The examples are presented with both jupyter notebooks and Python scripts.

![Querying with LTN](./docs/img/framework_querying.png)

![Learning with LTN](./docs/img/framework_learning.png)


## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/logictensornetworks/logictensornetworks/blob/master/LICENSE) file for details.

## Acknowledgements

LTN has been developed thanks to active contributions and discussions with the following people (in alphabetical order):
- Alessandro Daniele (FBK)
- Artur d’Avila Garcez (City)
- Benedikt Wagner (City)
- Emile van Krieken (VU Amsterdam)
- Francesco Giannini (UniSiena)
- Giuseppe Marra (UniSiena)
- Ivan Donadello (FBK)
- Lucas Bechberger (UniOsnabruck)
- Luciano Serafini (FBK)
- Marco Gori (UniSiena)
- Michael Spranger (Sony AI)
- Michelangelo Diligenti (UniSiena)
- Samy Badreddine (Sony AI)
- Sofoklis Kyriakopoulos (City)
