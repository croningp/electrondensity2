import logging

import argh

__logger__ = logging.getLogger(__name__)

from src.config import DATASETS, get_dataset

@argh.arg('dataset_name', choices=list(DATASETS.keys()))
def generate(dataset_name: str):
	"""
	Download a given datasets and generate electron densities from it.
    
	Args: 
		dataset_name: name of the dataset

	"""
	dataset = get_dataset(dataset_name)
	logging.info('Generating dataset {}'.format(dataset_name))
	dataset.generate()
	logging.info('Finished generating {} dataset'.format(dataset_name))

if __name__=='__main__':
	argh.dispatch_command(generate)
