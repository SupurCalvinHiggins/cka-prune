# Representation Aware Pruning with Centered Kernel Alignment

Some experiments with a new structured pruning algorithm based on centered kernel alignment. Created for my CSC 561 final project at the University of Rhode Island.

## Overview

### Algorithm

Centered kernel alignment (CKA) provides a measure of similiarity between neural network activations. CKA can be used to compute the relative "damage" caused by pruning a particular neuron via the following procedure.

For each neuron in the layer
1. Set the weights of the neuron to zero.
2. Compute the layer activations.
3. Compute the CKA score between the original and new activations.
4. Restore the weights of the neuron.

Initutively, high CKA scores correspond to low damage to network representations. Conversely, low CKA scores correspond to high damage to network representations. 

To prune the network, neurons are greedily removed from each layer according to the relative damage scores until all layers have been pruned by $p$%. The damages scores are recomputed after removing each neuron and the original activations are recomputed at the start of each pruning pass. This takes $\Omega(n^2)$ time where $n$ is the number of neurons in the layer.

### Repository

This repository contains the project report, presentation, data and scripts for reproducibility. The hyperparameter search, model training and pruning are all automated.

## Installation

### Python

This project requires Python 3.9.6 or Python 3.10.6. Other Python version might also work but have not been tested. See [here](https://www.python.org/downloads/) for details on installing Python 3.9.6 or Python 3.10.6.

### Python Dependencies

The Python dependencies are listed in **requirements.txt**. Installation instructions for these dependencies are given below.

Navigate to the root directory of the project and execute the following command.

```bash
pip install -r requirements.txt
```

For convenience, the **lib/** folder contains Google's implementation of CKA. The original source code can be found [here](https://github.com/google-research/google-research/tree/master/representation_similarity).

## Hyperparameter Search

To perform the learning rate hyperparameter search, call **main_search.py** with a configuration file defining the relevant model architecture and training configuration. For example, the following command performs the hyperparameter search for **config/ex1/cka.json** with WandB.
```bash
python3 main_search.py config/ex1/cka.json
```

To perform the same hyperparameter search with Seawulf, move **scripts/search.sh** into the root directory and execute the following command.
```bash
sbatch search.sh config/ex1/cka.json
```

## Training

To train the models, call **main_train.py** with a configuration file defining the relevant model architecture, training configuration and seeds. For example, the following command trains the models defined by **config/ex1/cka.json**.
```bash
python3 main_train.py config/ex1/cka.json
```

The resulting models will be saved to the **models/** folder.

To train the same models with Seawulf, move **scripts/train.sh** into the root directory and execute the following command.
```bash
sbatch train.sh config/ex1/cka.json
```

## Pruning

To prune the models, call **main_prune.py** with a configuration file defining the relevant model architecture, training configuration, pruning configuration and seeds. For example, the following command prunes the models defined by **config/ex1/cka.json**. Note that the models must already have been trained with **main_train.py**.
```bash
python3 main_prune.py config/ex1/cka.json
```

The resulting output data will be saved to the **output/** folder.

To prune the same models with Seawulf, move **scripts/prune.sh** into the root directory and execute the following command.
```bash
sbatch prune.sh config/ex1/cka.json
```

## References

\[1\] Jonathan Frankle and Michael Carbin. The Lottery Ticket Hypothesis:
Finding Sparse, Trainable Neural Networks. 2019. arXiv: 1803.03635 \[cs.LG\].

\[2\] Simon Kornblith et al. Similarity of Neural Network Representations Re-
visited. 2019. arXiv: 1905.00414 \[cs.LG\].
