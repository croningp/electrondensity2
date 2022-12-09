from abc import ABCMeta, abstractclassmethod
import os
from src import DATA_DIR

class Dataset(metaclass=ABCMeta):
    def __init__(self):
        super(Dataset, self).__init__()
        self.create_directories()
        
    def create_directories(self):
        if not os.path.exists(DATA_DIR):
            os.mkdir(DATA_DIR)
        if not os.path.exists(self.dir):
            os.mkdir(self.dir)
        
    @property
    @abstractclassmethod
    def name(self):
        raise NotImplementedError('Abstract class method')
    
    @property
    def dir(self) -> str:
        return os.path.join(DATA_DIR, self.name)
    
    @property
    def data_path(self) -> str:
        return os.path.join(self.dir, f'.{self.name}.tfrecords')
    
    def is_generated(self):
        return os.path.exists(self.data_path)
