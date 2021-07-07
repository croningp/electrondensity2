import numpy as np
import orbkit as ok

from ..utils.preprocessing import parse_log_file
from scipy.spatial.distance import cdist

class ESP():
    '''
    Class for calculating electrostatic potential grids from density grids.
    Density grids are read via orbkit from GAMESS calculation log files.
    The overall electrostatic potential grid is calculated numerically,
    solving for the potential in user-defined sub-grids.

    The potential is calculated numerically according to coulombs law:

              ESP = SUM_i(Z_i/R_i - r) - SUM_i(Q_i/r_i - r)

    Where r is a point on the grid, r_i is another point on the grid with
    charge Q_i and R_i is a neucleus with atomic number Z_i.

    Arguments
    ---------
        gamess_output_path: (str)
            path to GAMESS log file from calculation

        n_points: (int, default = 64)
            number of points on the grid along each axis

        step_size: (float, default = 0.625)
            distance (in Bohr) between grid points on a given axis
    '''

    def __init__(self, gamess_output_path, n_points=64, step_size=0.625):

        self.gamess_output_path = gamess_output_path
        self.density_grid = parse_log_file(self.gamess_output_path,
                                        lambda x: 0,
                                        n_points=n_points,
                                        step_size=step_size).density

        self.n_points = n_points - 1
        self.step_size = step_size
        self.grid_n = [n_points] * 3
        self.grid_min = [-n_points / 2 * step_size + step_size / 2] * 3
        self.grid_max = [n_points / 2 * step_size - step_size / 2] * 3


    def _generate_av_dens_grid(self, density_grid):
        '''
        Creates density grid that conserves number of electrons by
        averaging nearest neighbour points on grid and multiplying by
        grid segment volume
        '''

        x, y, z = np.array(density_grid.shape) - 1
        av_grid = np.zeros((x, y, z))

        lims = list(zip(range(0, density_grid.shape[0]-1),
                        range(2, density_grid.shape[0]+1)))

        for x1, x2 in lims:
            for y1, y2 in lims:
                for z1, z2 in lims:
                    grid_slice = density_grid[x1:x2,y1:y2,z1:z2]
                    av_grid[x1,y1,z1] = np.mean(grid_slice)*(self.step_size**3)

        return av_grid


    def _create_coord_array(self):
        '''
        Converts the grid to cartesian coordinates and returns these
        both in list and grid formats.
        '''
        coord = []
        for i in np.linspace(self.grid_min[0], self.grid_max[0], self.n_points):
            for j in np.linspace(self.grid_min[1], self.grid_max[1], self.n_points):
                for k in np.linspace(self.grid_min[2], self.grid_max[2], self.n_points):
                    coord.append([i, j, k])

        coord = np.array(coord)
        coord_grid = coord.reshape((self.av_grid.shape[0],
                                    self.av_grid.shape[1],
                                    self.av_grid.shape[2],
                                    3))
        return coord, coord_grid


    def _gen_coord_segments(self, coord_array, seg_ranges):
        '''
        Generates a grid segment for which esp is to be calculated at each
        point
        '''
        for x1, y1 in seg_ranges:
            for x2, y2 in seg_ranges:
                for x3, y3 in seg_ranges:
                    yield coord_array[x1:y1,x2:y2,x3:y3], x1, y1, x2, y2, x3, y3


    def _calculate_potential_segment(self, coord, coord_seg, coord_nuc, z_nuc):
        '''
        Calculates esp values within a given grid segment
        '''
        seg_x, seg_y, seg_z, _ = coord_seg.shape
        coord_seg = coord_seg.reshape((int(coord_seg.size/3), 3))

        # calc interaction with grid
        r_el = cdist(coord_seg, coord)
        q_el = self.av_grid.flatten()
        pots_el = np.divide(q_el, r_el)
        pots_el[pots_el == np.inf] = 0
        summed_el = np.sum(pots_el, axis=1)

        # calc interaction with nuclei
        r_nuc = cdist(coord_seg, coord_nuc)
        pots_nuc = np.divide(z_nuc, r_nuc)
        pots_nuc[pots_nuc == np.inf] = 0
        summed_nuc = np.sum(pots_nuc, axis=1)

        # calculate total potential
        tot_pot = summed_nuc - summed_el

        # reconstruct grid segment
        esp_seg = tot_pot.reshape((seg_x, seg_z, seg_z))

        return esp_seg


    def calculate_esp_grid(self, segment_size=3):
        '''
        Arguments
        ---------
            segment_size: (int, default=3)
                size of sub-grids to consider when calculating the esp grid.
                larger sub-grids are more memory intensive due to the storage
                of large euclidean distance matrices. Recommended value is
                max size that can be computed in memory.

        Returns
        -------
            esp_grid: (ndarray)
                NxNxN array containing total electrostatic potentials computed
                at each point in the grid
        '''

        # read info
        qc = ok.read.main_read(self.gamess_output_path, itype='gamess', all_mo=False)
        coord_nuc = qc.geo_spec

        # calculate effective loss of electrons and scale nuclear charges
        self.av_grid = self._generate_av_dens_grid(self.density_grid)
        n_electrons_true = np.sum(np.array(qc.geo_info[:,2]).astype(float))
        n_electrons_eff = np.sum(self.av_grid)
        n_electrons_frac = (n_electrons_eff/n_electrons_true)
        z_nuc = np.array(qc.geo_info[:,2]).astype(float)
        z_nuc_eff = z_nuc * n_electrons_frac

        # specify matrix segmentation
        seg_ranges = list(zip(range(0, self.n_points+1, segment_size),
                              range(segment_size, self.n_points+1, segment_size)))

        # calculate potential grid by segments
        esp_grid = np.zeros(self.av_grid.shape)
        coord, coord_grid = self._create_coord_array()
        for coord_seg in self._gen_coord_segments(coord_grid, seg_ranges):

            coord_seg, x1, y1, x2, y2, x3, y3 = coord_seg

            esp_seg = self._calculate_potential_segment(coord,
                                                        coord_seg,
                                                        coord_nuc,
                                                        z_nuc_eff)
            esp_grid[x1:y1,x2:y2,x3:y3] = esp_seg

        return esp_grid

