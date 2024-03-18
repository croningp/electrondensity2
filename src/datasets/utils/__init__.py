# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 12:22:51 2021

@author: jmg
"""
import os
import requests
import tarfile
import logging

from tqdm import tqdm

logger = logging.getLogger(__name__)

def download_and_unpack(url, destdir):
    
    # if os.path.exists(destdir):
    #     logger.info('Dataset already downloaded and unpacked')
    #     return

    r = requests.get(url, stream=True)
    total_size = int(r.headers.get('Content-Length'))
    file_name = r.url.split('?')[0].split('/')[-1]
    file_path = os.path.join(destdir, file_name)
    initial_pos = 0
    
    with open(file_path, 'wb') as file:
        with tqdm(total=total_size, unit='B', unit_scale=True,
                  desc=file_name, initial=initial_pos, ascii=True) as pbar:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
                    pbar.update(len(chunk))
                    
    if file_name.endswith('.tar.bz2'):
        with tarfile.open(file_path) as tar:
            for member in tqdm(iterable=tar.getmembers(), total=len(tar.getmembers()), 
                               desc='Extracting files', unit_scale=True):
                tar.extract(member, destdir)
    else:
        raise ValueError('Unsupported archive file {}'.format(file_name))
    os.remove(file_path)