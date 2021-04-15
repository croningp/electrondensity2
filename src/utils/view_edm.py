"""
Created on Thu Apr  9 11:29:39 2020. Displays the electron density using orbkit
and mayavi.

@author: jmg. Adapted now by Juan Manuel Parrilla (juanma@chem.gla.ac.uk)
"""

import pickle
import argparse

from orbkit import grid, output


def set_grid(n_points, step_size):
    """Sets and initilizes the grid of the orbkit.
    This works by modifying directly the grid modules variables.
    This function was originally in branch develop, and inside input.cube.
    I (me juanma) copied this function here so that I don't need to import the whole file.

    Args:
        n_points: int number of points in the grid cube along each face
        step_size: float specifies the grid spacing (Bohr)

    Returns:
        bbox: an array with grid boundaries for each axis
    """

    # set the grid for calculation
    grid.N_ = [n_points] * 3
    grid.min_ = [-n_points / 2 * step_size + step_size/2] * 3
    grid.max_ = [n_points / 2 * step_size - step_size/2] * 3
    grid.delta_ = [step_size] * 3


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="file to open", type=str)
    parser.add_argument("--render", help="EDMs to render", default=5, type=int)
    args = parser.parse_args()
    print('Opening {}'.format(args.file))

    set_grid(64, 0.5)
    grid.init_grid()

    with open(args.file, 'rb') as pfile:
        cubes = pickle.load(pfile)

    orig = cubes[0]  # original EDMs from the validation set
    gene = cubes[1]  # generated EDMs from the validation set. Related 1 to 1 to orig

    for i in range(args.render):
        output.view_with_mayavi(grid.x, grid.y, grid.z, orig[i, :, :, :, 0])
        output.view_with_mayavi(grid.x, grid.y, grid.z, gene[i, :, :, :, 0])
