# ElectronDensity2

### Setup

We recommend running Electrondensity in conda environment. The following command creates conda environment with necessary dependencies. 

`conda env create -f environment.yml`

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
```sh
python bin/datagen.py QM9Dataset
```
This command will download and generate electron densities for QM9 dataset (it requires 256GB od harddisk space) using number of CPUs specified in CPU_COUNT.
### Training the model
### Generating new electron densities