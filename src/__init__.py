import os
import multiprocessing
import logging

CPU_COUNT = os.environ.get('CPU_COUNT', -1)
if CPU_COUNT == -1:
	CPU_COUNT = multiprocessing.cpu_count()
    
DATA_DIR = os.environ.get('DATA_DIR', './data')
LOGS_DIR = os.environ.get('LOGS_DIR', './logs')
MODELS_DIR = os.environ.get('MODELS_DIR', './models')



