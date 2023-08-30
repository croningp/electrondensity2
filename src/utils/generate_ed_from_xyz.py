##########################################################################################
#
# This script was used to obtain the electron densities and electrostatic potentials
# from single XYZ files. We used it to generate the EDs and ESP from hosts.
# To generate the EDs from a lot of files, like for example the QM9 dataset, do not use
# this script, but the scripts we have in the Dataset folder.
#
# Author: Jarek & Juanma (juanma.parrilla@gcu.ac.uk)
#
##########################################################################################

import numpy as np
import shutil
import os
import pickle

from src.datasets.utils.esp import ESP
from src.datasets.utils.xtb import prepare_xtb_input, run_xtb
from src.datasets.utils.orbkit import electron_density_from_molden
import matplotlib.pyplot as plt



def load_xyz(file_path, read_energies=False):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        lines = [l.strip('\n') for l in lines]
    
    idx = 0
    confs = []
    energies = []
    while idx < len(lines):
        num_atoms = int(lines[idx].strip(' '))
        if read_energies:
            conf_energy = lines[idx+1].strip(' ')
            conf_energy = float(conf_energy)
            energies.append(conf_energy)
        idx += 2
        elements = []
        coords = []
        
        for lidx in range(num_atoms):
            parsed_line = [l for l in lines[lidx+idx].split(' ') if l!='']
            atom = parsed_line[0]
            atom_coords = [float(c) for c in parsed_line[1:]]
            elements.append(atom)
            coords.append(atom_coords)
        confs.append([elements, np.array(coords)])
        idx += num_atoms
    if not read_energies:
        return confs
    else:
        return confs, energies


if __name__ == '__main__':
    
    # these values should be the same as the ones you used to generate the dataset
    n_points = 64
    step_size = 0.5

    # class to calculate the electrostatic potentials
    esp = ESP(n_points, step_size)  

    # replace the file here with whatever host you want to convert
    output_dir = "cb6_output"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # replace the file here with whatever host you want to convert
    elements, coords = load_xyz('cb6v2.xyz')[0]
    # xtb_input_file_path = 'input.xtb'
    xtb_input_file_path = os.path.join(output_dir, 'input.xtb')
    
    # this is to fit the the prepare_xtb_input function
    coords = [[e]+list(c) for e, c in  zip(elements, coords)]
    
    prepare_xtb_input(coords, xtb_input_file_path)
    #xtb_exec_path = "/home/jarek/xtb-6.4.1/bin/xtb"
    xtb_exec_path = shutil.which('xtb')
    run_xtb(xtb_exec_path, xtb_input_file_path, output_dir, molden=True, esp=True)

    molden_input = os.path.join(output_dir, 'molden.input')
    rho = electron_density_from_molden(molden_input, n_points=n_points,
                                           step_size=step_size)
    
    # calculate esp cube from xtb
    espxtb_input = os.path.join(output_dir, 'xtb_esp.dat')
    molecule_esp = esp.calculate_espcube_from_xtb(espxtb_input)

    with open('cb6ED.pkl', 'wb') as file:
        pickle.dump(rho, file)

    with open('cb6ESP.pkl', 'wb') as file:
        pickle.dump(molecule_esp, file)
    
    # plt.imsave('test.png', np.tanh(rho[32]))

    


    





    
