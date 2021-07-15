##########################################################################################
# Created on Thu Apr  9 11:29:39 2020. Displays the electron density using orbkit
# and mayavi.
#
# @author: jmg. Adapted now by Juan Manuel Parrilla (juanma@chem.gla.ac.uk)
#
##########################################################################################

import pickle
import argparse

from orbkit import grid, output

from src.datasets.utils.orbkit import set_grid


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

        ed = cubes['electron_density']
        esp = cubes['electrostatic_potential']

        from scipy.ndimage import gaussian_filter, grey_dilation, uniform_filter
        espn = grey_dilation(esp*-1, size=3)
        espp = grey_dilation(esp, size=3)
        esp = espn*-1 + espp
        esp = gaussian_filter(esp, sigma=1)
        # esp = uniform_filter(esp, size=3)
        output.view_with_mayavi(grid.x, grid.y, grid.z, esp[:,:,:])
        #output.view_with_mayavi(grid.x, grid.y, grid.z, orig[i, :, :, :, 0])


    # if len(cubes) == 2:
    #     orig = cubes[0]  # original EDMs from the validation set
    #     gene = cubes[1]  # generated EDMs from the validation set. Related 1 to 1 to orig

    #     for i in range(args.render):
    #         output.view_with_mayavi(grid.x, grid.y, grid.z, orig[i, :, :, :, 0])
    #         output.view_with_mayavi(grid.x, grid.y, grid.z, gene[i, :, :, :, 0])

    # elif len(cubes) == 4:
    #     for i in range(args.render):
    #         output.view_with_mayavi(grid.x, grid.y, grid.z, cubes[0][i, :, :, :, 0])
    #         output.view_with_mayavi(grid.x, grid.y, grid.z, cubes[1][i, :, :, :, 0])
    #         output.view_with_mayavi(grid.x, grid.y, grid.z, cubes[2][i, :, :, :, 0])
    #         output.view_with_mayavi(grid.x, grid.y, grid.z, cubes[3][i, :, :, :, 0])
    # else:
    #     for i in range(args.render):
    #         output.view_with_mayavi(grid.x, grid.y, grid.z, cubes[i][:, :, :, 0])
