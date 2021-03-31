#CONDA_ENV_NAME=electrondensity
export PROJECT_ROOT=`pwd`
export DATA_DIR='./data'
export LOGS_DIR='./logs'
export MODELS_DIR='./models'
export CPU_COUNT=-1

#conda activate $CONDA_ENV_NAME

export ORBKITPATH=$PROJECT_ROOT/orbkit

if [ ! -d "$ORBKITPATH" ]
then
	echo "Couldn't find orbkit. Downloading and installing..."
	git clone https://github.com/orbkit/orbkit.git
	cd $ORBKITPATH
	git checkout cc17072
	python setup.py build_ext --inplace clean
	cd $PROJECT_ROOT
fi

export PYTHONPATH=$PYHONPATH:$ORBKITPATH


