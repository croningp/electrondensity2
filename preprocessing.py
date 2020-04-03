# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 15:02:00 2020

@author: jmg
"""


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
    coordinates = [c.split('\t')[:-1] for c in coordinates]
    frequencies = data[2+num_atoms].split('\t')
    smiles = ''.join(data[3+num_atoms].split('\t'))
    inchi = data[4+num_atoms]
     
    return num_atoms, prop_dict, coordinates, frequencies, smiles

def prepare_xtb_input(file_path, coordinates):
    
    coordinates = [c[1:]+[c[0]] for c in coordinates]
    with open(file_path, 'w+') as file:
        file.write('$coords angs\n')
        for c in coordinates:
            file.write('\t'.join(c)+'\n')
        file.write('$end\n')
        
        
    


if __name__ == '__main__':
    n, p, c, f, s = read_qm9_file('C:\\Users\\jmg\\Desktop\\programming\\electrondensity2\\data\\qm9\\dsgdb9nsd_000003.xyz')
    prepare_xtb_input('test.xtb', c)