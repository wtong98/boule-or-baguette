# Equality reasoning
This repository contains code that reproduces experiments from our paper [Boule or Baguette? A Study on Task Topology, Length Generalization, and the Benefit of Reasoning Traces](https://arxiv.org/abs/2602.14404).

## Installation

We use Python 3.10.12.

To install the requisite Python packages, run
```sh
pip install -r requirements.txt
```
By default, this will install CPU-only Jax. For GPU-enabled Jax, run
```sh
pip install -U "jax[cuda12]"
```
Depending on your machine, you may need additional or alternative packages for GPU-enabled Jax. Please consult the [Jax installation instructions](https://jax.readthedocs.io/en/latest/installation.html) for details.


## Running the code
Code organization:
* `experiment/`: experiments and plotting
* `model/`: model implementations
* `task/`: task implementations
* `train.py`: training routines
* `common.py`: shared utilities

Experiments are organized per file. For example, `16_scale_clean.py` will plot results on our simple transitive inference tasks. Each experiment file expects results to be present in `experiment/remote`, which contains scripts that generate the results, and are intended to be run on a compute cluster. Running the corresponding `run.py` script for each sub-experiment will generate the corresponding results.

Experiment files are formatted as [Jupyter code cells](https://code.visualstudio.com/docs/python/jupyter-support-py#_export-a-jupyter-notebook). One way of running these files interactively is through the [Jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) in VSCode. However, because the formatting is implemented through comment-based delimiters, these files may also be run as standard Python files.

If you notice any errors or issues, we welcome your pull requests!