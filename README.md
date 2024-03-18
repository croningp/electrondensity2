[![DOI](https://zenodo.org/badge/576255937.svg)](https://zenodo.org/doi/10.5281/zenodo.10409769)

# ElectronDensity2

This repository is related to the paper "Electron Density-Based GPT for Optimization and Suggestion of Host-Guest Binders". Please check the manuscript, and especially the Supplementary Information document, for full information about how to run the model, and about the results we obtained.

You can find the paper [here](https://www.nature.com/articles/s43588-024-00602-x). Within the paper you can also find the Supplementary Information document.

What follows is a very brief guide about how to get the repository started.

### Setup

We recommend running Electrondensity in conda environment. The following command creates conda environment with necessary dependencies. 

`conda env create -f environmentTF10.yml`

File `env.sh` contains environmental variables that can be modified to suit your configuration:

DATA_DIR is the path for storing datasets
LOG_DIR is the path for storing log files
MODEL_DIR is a path for storing and saving models
CPU_COUNT specifies number of cpus used for parallel processing of datasets default -1 means all available CPUs.

It also downloads and installs `orbkit` package which cannot be installed using conda. To finish installation and activate electrondensity environmet run:
```sh
source env.sh
```
### Generating training data

The following command might take days and use almost 300Gb. For a shorter test you can open [src/datasets/qm9.py, line 188](https://github.com/croningp/electrondensity2/blob/c729830924575cef7c73ea36a68814ca0b60793f/src/datasets/qm9.py#L188) (should be inside _compute_electron_density), and you can uncomment the list slice at the end.

```sh
python bin/generate_dataset.py QM9
```
This command will download and generate electron densities for QM9 dataset using number of CPUs specified in CPU_COUNT.
### Supplementary information
Please check the Supplementary Information document for full documentation about how to install, train and run the different models.
