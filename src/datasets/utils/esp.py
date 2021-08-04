import numpy as np
import orbkit as ok

from collections import namedtuple
from scipy.spatial.distance import cdist

from orbkit import read, grid, display, core

# from src.datasets.utils.orbkit import check_grid

Data = namedtuple(
    'Data', ('density', 'HOMO_LUMO_gap', 'n_points', 'step_size')
)

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
    grid.min_ = [-n_points / 2 * step_size + step_size/2] * 3
    grid.max_ = [n_points/2 * step_size - step_size/2] * 3
    grid.delta_ = [step_size] * 3


def check_grid(qc, n_points, step_size):
    """Checks if molecule will fit into the target grid

       Args:
           qc: orbkit object with parsed log file
           n_points: int number of points in the grid cube along each face
           step_size: float specifies the grid spacing (Bohr)

         Returns:
             boolean: True if the target_grid is bigger or equal molecular 
             box else False
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


def parse_log_file(path, property_fn,  n_points=64, step_size=0.625):
    """Parses a single file generating electron density which will be
    pickled to hardisk. Before this it checks if the molecule will fit
    the target grid.

    Args:
        path: a string to the gamess log file
    """
    # parse gamess log file
    qc = read.main_read(path, itype='molden', all_mo=False)
    # check if molecule fits into grid
    if check_grid(qc, n_points=n_points, step_size=step_size) == True:

        # calculate property
        prop = property_fn(path)
        # calculate electron density
        rho = core.rho_compute(qc, slice_length=1e4, numproc=16)

        data = Data(density=rho, HOMO_LUMO_gap=prop, n_points=step_size,
                    step_size=step_size)

        return data
    else:
        raise ValueError('Molecule doesn\'t fit into the target box')
    return None


class ESP:
    '''
    Class for calculating electrostatic potential grids from density grids.
    Density grids are read via orbkit from molden  log files.
    The overall electrostatic potential grid is calculated numerically,
    solving for the potential in user-defined sub-grids.

    The potential is calculated numerically according to coulombs law:

              ESP = SUM_i(Z_i/R_i - r) - SUM_i(Q_i/r_i - r)

    Where r is a point on the grid, r_i is another point on the grid with
    charge Q_i and R_i is a neucleus with atomic number Z_i.

    Arguments
    ---------
        n_points: (int, default = 64)
            number of points on the grid along each axis

        step_size: (float, default = 0.625)
            distance (in Bohr) between grid points on a given axis

    Methods
    --------
        calculate_esp_grid: Calculate ESP grid using specified
            grid params and molden input file from XTB. 
    '''

    def __init__(self, n_points=64, step_size=0.5):

        self.n_points = n_points
        self.step_size = step_size


    def calculate_espcube_from_xtb(self, esp_xtb):
        """ Given the electrstatic potential array built using xtb (option --esp), this
        function will place the sparse array into a cube.

        Args:
            esp_xtb: File as generated using "xtb --esp"

        Returns:
            cube with positions filled using data from the xtb array
        """

        # read xtb file, which contains sparse xyz and their charge
        data = np.genfromtxt(esp_xtb)
        # create canvas cube with all 0s
        cube = np.zeros((self.n_points,self.n_points,self.n_points))
        # use step size to see how big is a voxel
        factor = 1/self.step_size
        center = self.n_points//2

        for entry in data:
            x, y, z, e = entry
            # adjust coordinates to cube
            x = int(x*factor+center)
            y = int(y*factor+center)
            z = int(z*factor+center)
            cube[x, y, z] += e

        return cube