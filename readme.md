# Multipath Neural Network Experiments

This repository contains scripts to run the experiments described in the ICML2017 paper *[Deciding How to Decide: Dynamic Routing in Artificial Neural Networks](https://arxiv.org/abs/1703.06217)*, and visualize the results. All scripts are intended to be run from the root directory.

## Dependencies
- [Python3](https://www.python.org/downloads/), [Numpy](https://docs.scipy.org/doc/numpy/user/install.html), and [TensorFlow](https://www.tensorflow.org/install/) are required to train and test networks.
- [Matplotlib](http://matplotlib.org/users/installing.html) and [Seaborn](http://seaborn.pydata.org/installing.html) are required to generate figures.

## Library Modules
- `scripts/lib/data.py` defines the `Dataset` class that provides access to the datasets downloaded by `scripts/prep-data`, and implements data augmentation.
- `scripts/lib/layer_types.py` defines network layers that perform transformations and/or assign costs to network states.
- `scripts/lib/net_types.py` defines statically-routed, actor, and critic networks.
- `scripts/lib/desc.py` defines `net_desc`, a function that returns a serializable description of a network's structure and performance statistics, and `render_net_desc`, which returns a human-readable summary of this description.
- `scripts/lib/serdes.py` defines network serialization and deserialization functions.

## Experiment-Running Scripts
- `scripts/prep-data` downloads and formats MNIST, CIFAR-2, CIFAR-5, CIFAR-10, and the hybrid MNIST/CIFAR-10 dataset. The datasets are stored as `.npz` archives in the `data/` directory. It is necessary to run this script before running any others.
- `scripts/train-nets` trains and validates a set of networks. `scripts/train-nets --help` prints a list of available experiments, with names in the form *\<dataset\>-\<net-type\>[-\<modifications\>]*. *\<dataset\>* corresponds to the name of a file in the `data` directory (after running `scripts/prep-data`). *\<net-type\>* is either "sr", "ac", or "cr", indicating statically-routed, actor, or critic nets, respectively. *\<modifications\>* indicates how the network architecture or training procedure will be modified (see the paper for details). The trained network parameters and performance statistics are stored in the `nets/` directory.
- `scripts/train-adaptive-nets` is analogous to `scripts/train-nets`, except that it trains and validates a single network, with the ability to adapt to various costs of computation.
- `scripts/arch_and_hypers.py` is a module that defines the architecture and hyperparameters used in `scripts/train-nets` and `scripts/train-adaptive-nets`.

## Visualization Scripts
- `scripts/make-acc-eff-plots` writes accuracy-efficiency plots to the `figures/` directory, assuming the prerequisite experiments have been run.
- `scripts/make-nlds` writes node-link diagrams to the `figures/` directory, assuming the prerequisite experiments have been run.
- `scripts/make-routing-hists` writes routing histograms to the `figures/` directory, assuming the prerequisite experiments have been run.
- `scripts/make-pres-figs` generates relatively simple figures, designed to be displayed in a live presentation.
