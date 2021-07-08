import os
import subprocess as sub
import numpy as np

def prepare_xtb_input(coordinates, output_path):
    """
    Prepares xtb input file from xyz input. 
    Also arranges molecule so that its geometric center is as the
    beginning of coordinate system.

    Args:
        coordinates: array of arrays with strings of atoms symbols
                     and their coordinates with xyz like format
        output_path: string path to where create the input file
        
    Returns:
        None
    """
    
    coordinates = [c[1:]+[c[0]] for c in coordinates]
    coordinates_xyz = [c[:-1] for c in coordinates]
    coordinates_xyz = np.array([[float(f) for f in c] for c in coordinates_xyz])
    
    min_coords = np.min(coordinates_xyz, axis=0)
    max_coords = np.max(coordinates_xyz, axis=0)
    diff = min_coords + (max_coords-min_coords) / 2
    diff = diff.reshape([1, -1])
    mod_coords = coordinates_xyz - diff
    mod_coords = [ [str(f) for f in c] for c in mod_coords]
    
    new_coords = []
    for idx, c in enumerate(mod_coords):
        new_line = c
        new_line.append(coordinates[idx][-1])
        new_coords.append(new_line)

    with open(output_path, 'w+') as file:
        file.write('$coords angs\n')
        for c in new_coords:
            file.write('\t'.join(c)+'\n')
        file.write('$end\n')
        
def run_xtb(
    xtb_file: str,
    xyz_file: str,
    save_folder: str,
    molden: bool = False,
    esp: bool = False,
    opt: bool = False
):
    """Run XTB job on given xtb input file, saving output to a given
    save folder.

    Args:
        xtb_file (str): path to xtb executable
        xyz_file (str): xtb input file to run geometry optimisation on.
        save_folder (str): Folder to save XTB output files to.
        molden (bool): If True will generate molden input file.
        opt (bool): If true will perform geomtry optmizatin.

    """
    os.makedirs(save_folder, exist_ok=True)
    cmd = [os.path.abspath(xtb_file), os.path.abspath(xyz_file)]
    if molden:
        cmd.append('--molden')
    if esp:
        cmd.append('--esp')
    if opt:
        cmd.append('--opt')
    sub.Popen(
        cmd,
        cwd=save_folder,
        stdout=sub.PIPE,
        stderr=sub.PIPE,
    ).communicate() 

