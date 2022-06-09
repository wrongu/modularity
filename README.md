Modularity by Inputs and Outputs
-----------------

This is the code repository accompanying the paper

> Lange, R. D., Rolnick, D. S., and Kording, K. (2022) "Clustering units in neural networks: upstream vs downstream information." TMLR. https://openreview.net/forum?id=Euf7KofunK 

## High-level structure

### 1. Model definitions and training

PyTorch models are defined in `models/mnist.py` and `models/cifar10.py`.  Models are wrapped by a Pytorch Lightning
module `models.LitWrapper`, which handles loading a specific model or dataset. Training is done by `train.py`, which
is called for a range of hyperparameter configurations by `train.sh`.

Training needs to be run before moving on to step 2.

### 2. Computing modularity

As detailed in the paper, we analyze "modularity" of a set of units (e.g. all units in a layer) by

1. computing pairwise similarity scores of units
2. clustering units together by maximizing the Q score from [Newman (2006)](https://doi.org/10.1073/pnas.0601602103).

Step 1 is done by functions in `associations.py` and step 2 is done by functions in `modularity.py`.

Running `eval.py` does the following:

- loads a model from a checkpoint
- computes a variety of performance statistics such as validation accuracy, weight norms, etc
- computes a variety of modularity statistics by calling functions from `associations.py` and `modularity.py`
- saves results back into the same checkpoint file

The file `eval.sh` is a shell script that demonstrates how we call `eval.py` for each checkpoint in a directory. 

### 3. Loading and plotting results

As mentioned above, `eval.py` loads a checkpoint, computes a variety of statistics including modules (clusters), and
saves the result back into the checkpoint file. This means that `eval.sh` needs to be run on a set of checkpoints before
notebooks can be run to plot the results. The file `analysis.py` handles the process of loading statistics computed by 
`eval.py` into a pandas DataFrame.

The notebook `notebooks/analysis_sandbox.ipynb` was used to generate most figures in the paper. This notebook's structure
primarily involves calling `analysis.load_data_as_table()` to load precomputed information from a set of checkpoints 
into a DataFrame, then the rest is a variety of ways of slicing and plotting the results.
