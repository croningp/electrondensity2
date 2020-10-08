# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 11:29:39 2020

@author: jmg
"""
import os
import numpy as np
import pickle


from orbkit import grid, output
#os.chdir('/home/jarek/electrondensity2')
from input.cube import set_grid

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("file", help="file to open",
                    type=str)
args = parser.parse_args()
print('Opening {}'.format(args.file))






set_grid(64, 0.5)
grid.init_grid()


with open(args.file, 'rb') as pfile:
    cubes = pickle.load(pfile)
print('input shape')
print(cubes.shape) 
    
#cubes = np.exp(-cubes)
#cubes -= 1e-4

if cubes.shape[-1] == 1:
    for i in range(5):
        output.view_with_mayavi(grid.x, grid.y, grid.z, cubes[i, :, :, :, 0])
else:
    for i in range(5):
        output.view_with_mayavi(grid.x, grid.y, grid.z, cubes[i, :, :, :, 0])
    for i in range(5):
        output.view_with_mayavi(grid.x, grid.y, grid.z, cubes[i, :, :, :, 1])

print('max', np.max(cubes[0]))
print('mean', np.mean(cubes[0]))


print('max', np.max(cubes[1]))
print('mean', np.mean(cubes[1]))