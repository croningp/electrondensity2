# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 15:02:00 2020

@author: jmg
"""
import os
import subprocess as sub
import pickle
import numpy as np
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
import tqdm
from electrondensity2.input.cube import parse_molden_file

def read_qm9_file(path):
    """Parses a single file from qm9 database
        https://springernature.figshare.com/collections/Quantum_chemistry_str\\
        uctures_and_properties_of_134_kilo_molecules/978904/5
        
        I.  Property  Unit         Description
        --  --------  -----------  --------------
        1  tag       -            "gdb9"; string constant to ease extraction via grep
        2  index     -            Consecutive, 1-based integer identifier of molecule
        3  A         GHz          Rotational constant A
        4  B         GHz          Rotational constant B
        5  C         GHz          Rotational constant C
        6  mu        Debye        Dipole moment
        7  alpha     Bohr^3       Isotropic polarizability
        8  homo      Hartree      Energy of Highest occupied molecular orbital (HOMO)
        9  lumo      Hartree      Energy of Lowest occupied molecular orbital (LUMO)
        10  gap       Hartree      Gap, difference between LUMO and HOMO
        11  r2        Bohr^2       Electronic spatial extent
        12  zpve      Hartree      Zero point vibrational energy
        13  U0        Hartree      Internal energy at 0 K
        14  U         Hartree      Internal energy at 298.15 K
        15  H         Hartree      Enthalpy at 298.15 K
        16  G         Hartree      Free energy at 298.15 K
        17  Cv        cal/(mol K)  Heat capacity at 298.15 K
        
    """
    with open(path) as qm9_file:
        data = qm9_file.readlines()
    data = [line.strip('\n') for line in data]
    num_atoms = int(data[0])
    properties = data[1].split('\t')[:-1]
    dict_keys = ['tag_index', 'A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo',
                 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']
    prop_dict = {}
    
    for idx, prop_name in enumerate(dict_keys):
        prop_dict[prop_name] = properties[idx]
    
    coordinates = data[2:2+num_atoms]
    coordinates = [c.replace('*^', 'e') for c in coordinates]
    coordinates = [c.split('\t')[:-1] for c in coordinates]
    frequencies = data[2+num_atoms].split('\t')
    smiles = ''.join(data[3+num_atoms].split('\t')[0])
    inchi = data[4+num_atoms]
     
    return num_atoms, prop_dict, coordinates, frequencies, smiles

def prepare_xtb_input(file_path, coordinates):
    
    coordinates = [c[1:]+[c[0]] for c in coordinates]
    coordinates_xyz = [c[:-1] for c in coordinates]
    coordinates_xyz = np.array([[float(f) for f in c] for c in coordinates_xyz])
    
    min_coords = np.min(coordinates_xyz, axis=0)
    max_coords = np.max(coordinates_xyz, axis=0)
    diff = min_coords + (max_coords-min_coords) / 2
    diff = diff.reshape([1, -1])
    print(coordinates_xyz)
    print(diff)
    print(coordinates_xyz-diff)
    mod_coords = coordinates_xyz - diff
    mod_coords = [ [str(f) for f in c] for c in mod_coords]
    
    new_coords = []
    for idx, c in enumerate(mod_coords):
        new_line = c
        new_line.append(coordinates[idx][-1])
        new_coords.append(new_line)

    #print(new_coords)

    with open(file_path, 'w+') as file:
        file.write('$coords angs\n')
        for c in new_coords:
            file.write('\t'.join(c)+'\n')
        file.write('$end\n')
        
        
def run_xtb(
    xtb_file: str,
    xyz_file: str,
    save_folder: str,
    molden: bool = False,
    opt: bool = False
):
    """Run XTB geometry optimisation on given XYZ file, saving output to given
    save folder.

    Args:
        xtb_file (str): path to xtb executable
        xyz_file (str): XYZ file to run geometry optimisation on.
        save_folder (str): Folder to save XTB output files to.
        molden (bool): If Triue will generate molden input file.
        opt (bool): If true will perform geomtry optmizatin.

    """
    os.makedirs(save_folder, exist_ok=True)
    cmd = [os.path.abspath(xtb_file), os.path.abspath(xyz_file)]
    if molden:
        cmd.append('--molden')
    if opt:
        cmd.append('--opt')
    sub.Popen(
        cmd,
        cwd=save_folder,
        stdout=sub.PIPE,
        stderr=sub.PIPE,
    ).communicate() 


def parse_single_qm9_file(
            input_path: str,
            output_dir:str):
    """Computes electron desnity for a single qm9 file and pickle it along properties
        extracted from file and smiles string of the molecule.
    
        Args:
            input_path (str): qm9 file to extract properties and molecular geometry
            out_dir (str): directory where output files will be stored
    
    """
    num_atoms, properties, coords, _, smiles =  read_qm9_file(input_path)
    os.makedirs(output_dir, exist_ok=True)
    xyz_file = os.path.join(output_dir,'input.xtb' )
    prepare_xtb_input(xyz_file, coords)
    output_dir = os.path.abspath(output_dir)
    xtb_path = '/home/jarek/miniconda3/bin/xtb'
    run_xtb(xtb_path, xyz_file, output_dir, molden=True)
    molden_input = os.path.join(output_dir, 'molden.input')
    rho = parse_molden_file(molden_input, step_size=0.5 )
    output_dict = {}
    output_dict['electron_density'] = rho
    output_dict['properties'] = properties
    output_dict['smiles'] = smiles
    output_file = os.path.join(output_dir, 'output.pkl')
    with open(output_file, 'wb+') as ofile:
        pickle.dump(output_dict, ofile)

def parse_dataset(dataset_path, result_dir):
    files = os.listdir(dataset_path)
    os.makedirs(result_dir, exist_ok=True)
    for f in tqdm.tqdm(files):
        file_path = os.path.join(dataset_path, f)
        file_id = f.strip('.xyz').split('_')[1]
        output_dir = os.path.join(result_dir, file_id)
        parse_single_qm9_file(file_path, output_dir)
    
if __name__ == '__main__':
    parse_dataset('data/', '/media/extssd/jarek/qm9_cubes') 
   # source_path = 'C:\\Users\\jmg\\Desktop\\programming\\electrondensity2_testing\\data\\qm9'
   # out_dir = 'D:\\qm9'
   # folders = os.listdir(out_dir)
   # for f in tqdm.tqdm(folders):
   #     f_path = os.path.join(out_dir, f, 'output.pkl')
   #     print(f_path)
   #     with open(f_path, 'rb') as dfile:
    #     data = pickle.load(dfile)
    #    
    #    index = data['properties']['tag_index'].split(' ')[1]
    #    source_name = 'dsgdb9nsd_{:06d}.xyz'.format(int(index))
    #    file_path = os.path.join(source_path, source_name)
        
    #    original_data = read_qm9_file(file_path)
    #    assert original_data[1]['tag_index'] == data['properties']['tag_index']
        
    #    data['smiles'] = original_data[4]
     #   with open(f_path, 'wb') as dfile:
      #      pickle.dump(data, dfile)
            
            
        
        


