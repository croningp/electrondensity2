# ElectronDensity2

### Setup

We recommend running electrondesity in conda environment

`conda env create -f environment.yml`

creates conda environment with dependencies recommended to
run experiments described in the paper.

To activate conda environment run 

`source env.sh`

script env.sh contains environmental variables:
DATA_DIR is the path for storing datasets
LOG_DIR is the path for storing log files
MODEL_DIR is a path for storing and saving models
CPU_COUNT specifies number of cpus used for parallel processing of datasets

### generating training data
```
python bin/datagen.py qm9
```
This command will download and generate electron densities for QM9 dataset (it requires 256GB od harddisk space) using number of CPUs specified in CPU_COUNT.
### Training the model
### Generating new electron densities