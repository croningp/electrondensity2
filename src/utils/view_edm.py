##########################################################################################
# Created on Thu Apr  9 11:29:39 2020. Displays the electron density using orbkit
# and mayavi.
#
# @author: jmg. Adapted now by Juan Manuel Parrilla (juanma@chem.gla.ac.uk)
#
##########################################################################################

import pickle
import argparse
from tensorflow.nn import max_pool3d
import numpy as np

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

    # with open("pdcage.p", 'rb') as pfile:
    #     host = pickle.load(pfile)[0, 8:-8, 8:-8, 8:-8, 0] * -1
    #     # host = pickle.load(pfile)[8:-8, 8:-8, 8:-8]
    #     # output.view_with_mayavi(grid.x, grid.y, grid.z, host)

    # with open("cc6_esp.pkl", 'rb') as pfile:
    #     host = pickle.load(pfile) #* -1

    with open("cb6ESP.pkl", 'rb') as pfile:
        host = pickle.load(pfile) # * -1
        # output.view_with_mayavi(grid.x, grid.y, grid.z, host)

    # with open("pdcage_esp.pkl", 'rb') as pfile:
    #     host = pickle.load(pfile)[8:-8, 8:-8, 8:-8] #* -1

    # host = np.float32(host)
    # host = np.expand_dims(host, axis=[0, -1])
    # output.view_with_mayavi(grid.x, grid.y, grid.z, host[0,:,:,:,0])
    # datap = max_pool3d(host, 5, 1, 'SAME')
    # datan = max_pool3d(host*-1, 5, 1, 'SAME')
    # host = datap*1 + datan*-1
    # output.view_with_mayavi(grid.x, grid.y, grid.z, host[0, :, :, :, 0])
    # datap = max_pool3d(host, 3, 1, 'SAME')
    # output.view_with_mayavi(grid.x, grid.y, grid.z, datap[0, :, :, :, 0])

    # with open("cage_esp_optimized_final_ESP.p", 'rb') as pfile:
    #     esp = pickle.load(pfile)
    #     datap = max_pool3d(esp, 3, 1, 'SAME')
    #     datan = max_pool3d(esp*-1, 3, 1, 'SAME')
    #     esp = datap + datan*-1
    #     esp = esp / 0.33
    

    if len(cubes) == 2:
        orig = cubes[0]  # original EDMs from the validation set
        gene = cubes[1]  # generated EDMs from the validation set. Related 1 to 1 to orig

        for i in range(args.render):
            output.view_with_mayavi(grid.x, grid.y, grid.z, orig[i, :, :, :, 0])
            output.view_with_mayavi(grid.x, grid.y, grid.z, gene[i, :, :, :, 0])

    elif len(cubes) == 3:
        # prepare original esp for visualization
        datap = max_pool3d(cubes[1], 5, 1, 'SAME')
        datan = max_pool3d(cubes[1]*-1, 5, 1, 'SAME')
        origesps = datap + datan*-1

        for i in range(args.render):
            print(i+1)
            ed = cubes[0][i, :, :, :, 0]
            geneesp = cubes[2][i, :, :, :, 0]
            origesp = origesps[i, :, :, :, 0]

            output.view_with_mayavi(grid.x, grid.y, grid.z, ed)
            output.view_with_mayavi(grid.x, grid.y, grid.z, origesp)
            output.view_with_mayavi(grid.x, grid.y, grid.z, geneesp)
    else:
        for i in range(args.render):
            print(i+1)
            # output.view_with_mayavi(grid.x, grid.y, grid.z, esp[1]*cubes[1][:, :, :, 0])
            output.view_with_mayavi(grid.x, grid.y, grid.z, cubes[i][:, :, :, 0])
            output.view_with_mayavi(grid.x, grid.y, grid.z, host+cubes[i][:, :, :, 0])
