from src.datasets.qm9 import QM9Dataset

DATASETS = {'QM9': lambda: QM9Dataset(n_points=64, step_size=0.5)}

def get_dataset(name):
    if name not in DATASETS.keys():
        raise ValueError('Dataset {} does not exists'.format(name))
    return DATASETS[name]()
