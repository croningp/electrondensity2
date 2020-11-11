import os
import pickle
from collections import namedtuple

import tqdm
import numpy as np

from orbkit import read, grid, extras, output, display, core


Data = namedtuple('Data', ('density', 'HOMO_LUMO_gap', 'n_points',
                           'step_size', 'smiles'))


def check_grid(qc, n_points, step_size):
    """Checks if molecule will fit into the target grid

       Args:
           qc: orbkit object with parsed log file
           n_points: int number of points in the grid cube along each face
           step_size: float specifies the grid spacing (Bohr)

         Returns:
             boolean: True if the target_grid is bigger or equal molecular box
                      else False
    """
    grid.adjust_to_geo(qc, extend=2.0, step=step_size)
    grid.grid_init(force=True)
    display.display(grid.get_grid())
    molecule_bbox = grid.get_bbox()

    set_grid(n_points=n_points, step_size=step_size)
    grid.grid_init(force=True)
    display.display(grid.get_grid())
    target_bbox = grid.get_bbox()

    # reshape for each axis
    molecule_bbox = molecule_bbox.reshape((-1, 2))
    target_bbox = target_bbox.reshape((-1, 2))
    # for each dimension
    for i in range(3):
        if molecule_bbox[i][0] < target_bbox[i][0]:
            return False
        if molecule_bbox[i][1] > target_bbox[i][1]:
            return False
    return True

def set_grid(n_points, step_size):
    """Sets and initilizes the grid of the orbkit
    this works by modifying directly the grid modules variables.

    Args:
        n_points: int number of points in the grid cube along each face
        step_size: float specifies the grid spacing (Bohr)

    Returns:
        bbox: an array with grid boundaries for each axis
    """

    # set the grid for calculation
    grid.N_ = [n_points] * 3
    grid.min_ = [-n_points / 2 * step_size +step_size/2] * 3
    grid.max_ = [n_points/2* step_size -step_size/2] * 3
    grid.delta_ = [step_size] * 3

def parse_log_file(path, property_fn,  n_points=64, step_size=0.625):
    """Parses a single file generating electron density which will be
    pickled to hardisk. Before this it checks if the molecule will fit
    the target grid.

    Args:
        path: a string to the gamess log file
    """
        # parse gamess log file
    qc = read.main_read(path, itype='gamess', all_mo=False)
     # check if molecule fits into grid
    if check_grid(qc, n_points=n_points, step_size=step_size) == True:

        #calculate property
        prop = property_fn(path)
        # calculate electron density
        rho = core.rho_compute(qc, slice_length=1e4, numproc=27)

#        data = Data(density=rho, HOMO_LUMO_gap=prop, n_points=step_size,
#                         step_size=step_size)

        return rho, prop
    else:
        raise ValueError('Molecule doesn\'t fit into the target box')
    return None

def parse_molden_file(path, n_points=64, step_size=0.625):
    """Parses a single file generating electron density which will be
    pickled to hardisk. Before this it checks if the molecule will fit
    the target grid.

    Args:
        path: a string to the molden input file
    """
    # parse molden input file
    qc = read.main_read(path, itype='molden', all_mo=False)

     # check if molecule fits into grid
    if check_grid(qc, n_points=n_points, step_size=step_size) == True:

        # calculate electron density
        return core.rho_compute(qc, slice_length=1e4, numproc=1)

    else:
        raise ValueError('Molecule doesn\'t fit into the target box')
    return None

def coarse_grain(rho, size=4):
    """
    Reduces the dimensionality of the grid as described in arXiv:1809.02723 by
    summation  of  the density  values  in  non-overlapping cubes of size x size x size

    Args:
        rho: an array of shape [cube_shape, cube_shape, cube_shape] to be coarse grained
        size: int the size of non-overlapping cube
    Returns:
        rho: an array with cube of reduced dimensinality of shape
             [cube_shape /size, cube_shape /size, cube_shape /size]

    """


    block_shape = (size,) * len(rho.shape)

    new_shape = tuple(rho.shape// np.array(block_shape)) + block_shape
    new_strides = tuple(rho.strides * np.array(block_shape)) + rho.strides
    blocks =  np.lib.stride_tricks.as_strided(rho, shape=new_shape, strides=new_strides)
    rho = np.sum(blocks, axis=(3,4,5))

    return rho


def prepare_xtb_from_xyz(file_path, output_file):
    """
    Prepares xtb input file from xyz file and also centeres the molecule
    at the begining of coordinate system.
    
    Args:
        file_path: string path to xyz file 
        output_file: string path of the output file  
        

    """
    with open(file_path, 'r') as file:
        data = file.readlines()
    
    data = [d.strip('\n') for d in data]
    data = [d.split(' ') for d in data]
    data = [[c for c in d if c !=''] for d in data]
    atoms = [d[0] for d in data]
    coords = [d[1:] for d in data]
    coords = [[float(c) for c in d] for d in coords]
    coordinates_xyz = np.array(coords)
    print(atoms)
    print(coordinates_xyz)
    
    min_coords = np.min(coordinates_xyz, axis=0)
    max_coords = np.max(coordinates_xyz, axis=0)
    diff = min_coords + (max_coords-min_coords) / 2
    diff = diff.reshape([1, -1])

    mod_coords = coordinates_xyz - diff
    mod_coords = [ [str(f) for f in c] for c in mod_coords]
    
    new_coords = []
    for idx, c in enumerate(mod_coords):
        new_line = c
        new_line.append(atoms[idx])
        new_coords.append(new_line)

    #print(new_coords)

    with open(output_file, 'w+') as file:
        file.write('$coords angs\n')
        for c in new_coords:
            file.write('\t'.join(c)+'\n')
        file.write('$end\n') 
