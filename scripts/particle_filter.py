import numpy as np
import cv2
import math
from params import RosParams

from occupancy_grid import LocalOccupancyGrid

class ParticleFilterParams(RosParams):

    _params = [('dist_mu', 0.0, float, 'dist_mu'),  # this should be implicitly 0, it shouldn't even be a parameter
               ('dist_stdev', 1.6, float, 'dist_stdev'),
               ('max_cell_particles', 50, long, 'max_cell_particles'),
               ('max_particle_velocity', 25 * 0.1, float, 'max_particle_velocity'), # meters per second, 0.1 is the frame period
               ('new_particle_ratio', 0.2, float, 'new_particle_ratio'),
               ('min_particle_survival_probability', 0.5, float, 'min_particle_survival_probability'),
               ('min_particle_age', 0, int, 'min_particle_age'), # minimum age to use particle for dynamic and static probability ocmputation, should always be 0
               ('static_particle_ratio', 0.15, float, 'static_particle_ratio'),
               ('grid_occupied_threshold', 0.7, float, 'grid_occupied_threshold'),
               ('static_threshold', 0.01 * 0.1, float, 'static_threshold'), # minimum speed to consider particle as in motion, m/frame
               ]


class ParticleFilter(object):


    P_X = 0
    P_Y = 1
    P_Z = 2
    P_DX = 3
    P_DY = 4
    P_DZ = 5
    P_W = 6 # weight
    P_A = 7 # age


    # maximum at 1.0, not a pdf
    def gaussian(self, x, mu, sig):
        return  np.exp(- ( (x - mu) * (x - mu) / (2 * sig*sig)))

    def __init__(self, params):
        """
        :param LocalOccupancyGrid grid:
        """
        super(ParticleFilter, self).__init__()

        self.params = params

        self._grid = None
        """:type : LocalOccupancyGrid"""

        self._particles = np.empty((0, 8))

        self._cell_weights = np.empty((0, 0))

        self._particle_counts = None

    def update_grid(self, grid):
        self._grid = grid
        """:type : LocalOccupancyGrid"""

        self._particle_counts = None
        self._cell_weights = np.zeros(shape=(self._grid.rows(), self._grid.cols()))

    def step_model(self):
        # move particles
        self._particles[:, [self.P_X, self.P_Y, self.P_Z]] += self._particles[:, [self.P_DX, self.P_DY, self.P_DZ]]


        self._particles[:, self.P_A] += 1

        generated = self.sample_new()

        self._particles = np.vstack((self._particles, generated))

        # remove particles outside grid

        particles_in_grid = self._grid.in_range(self._particles[:, [self.P_X, self.P_Y, self.P_Z]])

        self._particles = self._particles[particles_in_grid, :]


        #remove overpopulation
        particle_counts = self.particle_counts()

        cell_ratio = self.params.max_cell_particles / np.maximum(1.0, particle_counts.astype(np.float))  # prevent div by zero

        rows = self._grid.get_row_i(self._particles[:, [self.P_X, self.P_Y, self.P_Z]])
        cols = self._grid.get_col_i(self._particles[:, [self.P_X, self.P_Y, self.P_Z]])

        keep_particle = np.random.sample(size=self._particles.shape[0]) < cell_ratio[rows, cols]

        self._particles = self._particles[keep_particle, :]
        self._particle_counts = None

    @staticmethod
    def cart2pol(x, y):
        theta = np.arctan2(y, x)
        rho = np.hypot(x, y)
        return theta, rho

    @staticmethod
    def pol2cart(theta, rho):
        x = rho * np.cos(theta)
        y = rho * np.sin(theta)
        return x, y


    def grid_angle_stdev(self, particles):

        #cicular stdev formula from https://github.com/scipy/scipy/blob/v0.19.0/scipy/stats/morestats.py#L2773-L2806
        # see also http://math.stackexchange.com/questions/2154174/calculating-the-standard-deviation-of-a-circular-quantity/2154190

        particle_counts = self.particle_counts()

        # for each cell separately, then weigh by number of items in cell
        cols = self._grid.cols()
        rows = self._grid.rows()

        p_rows = self._grid.get_row_i(particles[:, [self.P_X, self.P_Y, self.P_Z]])
        p_cols = self._grid.get_col_i(particles[:, [self.P_X, self.P_Y, self.P_Z]])


        particle_v = np.linalg.norm(particles[:, [self.P_DX, self.P_DY]], ord=None, axis=1) # frobenius same as p2 norm (Euclidean distance)
        particle_is_dynamic = (particle_v >= self.params.static_threshold).astype(np.float32)

        particles_theta_sin = np.sin(np.arctan2(particles[:, self.P_DY], particles[:, self.P_DX])) * particle_is_dynamic

        sum_theta_sin, _xedges, _yedges = np.histogram2d(p_rows, p_cols, bins=[rows, cols], range=[[0, rows - 1], [0, cols - 1]], normed=False, weights=particles_theta_sin)

        particles_theta_cos = np.cos(np.arctan2(particles[:, self.P_DY], particles[:, self.P_DX])) * particle_is_dynamic

        sum_theta_cos, _xedges, _yedges = np.histogram2d(p_rows, p_cols, bins=[rows, cols], range=[[0, rows - 1], [0, cols - 1]], normed=False, weights=particles_theta_cos)

        with np.errstate(divide='ignore', invalid='ignore'):
            avg_theta_sin = np.where(particle_counts == 0.0, np.nan, sum_theta_sin / particle_counts)

            avg_theta_cos = np.where(particle_counts == 0.0, np.nan, sum_theta_cos / particle_counts)

            R = np.hypot(avg_theta_sin, avg_theta_cos)

            #return np.where(np.logical_or(np.isnan(R), R == 0.0), np.nan, np.sqrt(-2.0 * np.log(R)))
            return np.where(np.logical_or(np.isnan(R), R == 0.0), np.nan, 1.0 - R)


    def get_masses(self):

        # for each cell separately, then weigh by number of items in cell
        cols = self._grid.cols()
        rows = self._grid.rows()
        
        min_age = self.params.min_particle_age

        particles = self._particles[self._particles[:, self.P_A] >= min_age]

        p_rows = self._grid.get_row_i(particles[:, [self.P_X, self.P_Y, self.P_Z]])
        p_cols = self._grid.get_col_i(particles[:, [self.P_X, self.P_Y, self.P_Z]])



        particle_v = np.linalg.norm(particles[:, [self.P_DX, self.P_DY]], ord=None, axis=1) # frobenius same as p2 norm (Euclidean distance)

        particle_is_static = (particle_v < self.params.static_threshold).astype(np.float32)

        static_particle_cnts, _xedges, _yedges = np.histogram2d(p_rows, p_cols, bins=[rows, cols], range=[[0, rows - 1], [0, cols - 1]], normed=False, weights=particle_is_static)

        dynamic_particle_cnts = self.params.max_cell_particles - static_particle_cnts

        mass_static = np.nan_to_num(static_particle_cnts / self.params.max_cell_particles)

        orientation_stdev = self.grid_angle_stdev(particles)

        orientation_stdev_max = np.nanmax(orientation_stdev)

        orientation_stdev_max = 1.0

        print("Orientation stdev_max: {0}".format(orientation_stdev_max))

        with np.errstate(divide='ignore', invalid='ignore'):
            mass_dynamic = np.nan_to_num((1.0 - orientation_stdev / orientation_stdev_max) * dynamic_particle_cnts / self.params.max_cell_particles)

        return mass_static, mass_dynamic

    def get_orientation_means(self):
        # for each cell separately, then weigh by number of items in cell
        cols = self._grid.cols()
        rows = self._grid.rows()

        p_rows = self._grid.get_row_i(self._particles[:, [self.P_X, self.P_Y, self.P_Z]])
        p_cols = self._grid.get_col_i(self._particles[:, [self.P_X, self.P_Y, self.P_Z]])

        particle_v = np.linalg.norm(self._particles[:, [self.P_DX, self.P_DY]], ord=None, axis=1) # frobenius same as p2 norm (Euclidean distance)

        particle_is_dynamic = (particle_v >= self.params.static_threshold).astype(np.float32)


        particles_theta_sin = np.sin(np.arctan2(self._particles[:, self.P_DY], self._particles[:, self.P_DX])) * particle_is_dynamic


        sum_theta_sin, _xedges, _yedges = np.histogram2d(p_rows, p_cols, bins=[rows, cols], range=[[0, rows - 1], [0, cols - 1]], normed=False, weights=particles_theta_sin)

        particles_theta_cos = np.cos(np.arctan2(self._particles[:, self.P_DY], self._particles[:, self.P_DX])) * particle_is_dynamic


        sum_theta_cos, _xedges, _yedges = np.histogram2d(p_rows, p_cols, bins=[rows, cols], range=[[0, rows - 1], [0, cols - 1]], normed=False, weights=particles_theta_cos)

        at = np.arctan2(sum_theta_sin, sum_theta_cos)

        return np.where(at > 0, at, 2.0 * np.pi + at)

    def distance_gauss_occupied(self):

        inv = (1 - self._grid.get_occupied_bin())

        distance = np.empty(self._grid.get_grid().shape, dtype=np.float32)

        cv2.distanceTransform(inv, cv2.DIST_L2, cv2.DIST_MASK_PRECISE, distance)
        return self.gaussian(distance, self.params.dist_mu, self.params.dist_stdev)


    def distance_gauss_free(self):


        distance = np.empty(self._grid.get_grid().shape, dtype=np.float32)

        cv2.distanceTransform(self._grid.get_occupied_bin(), cv2.DIST_L2, cv2.DIST_MASK_PRECISE, distance)
        return self.gaussian(distance, self.params.dist_mu, self.params.dist_stdev)



    def particle_counts(self):

        if self._particle_counts is None:
            # for each cell separately, then weigh by number of items in cell
            cols = self._grid.cols()
            rows = self._grid.rows()

            p_rows = self._grid.get_row_i(self._particles[:, [self.P_X, self.P_Y, self.P_Z]])
            p_cols = self._grid.get_col_i(self._particles[:, [self.P_X, self.P_Y, self.P_Z]])

            self._particle_counts, _xedges, _yedges = np.histogram2d(p_rows,
                                                                     p_cols,
                                                                     bins=[rows, cols],
                                                                     range=[[0, rows - 1], [0, cols - 1]])
        return self._particle_counts

    def calc_cell_weights(self):


        dist_occupied = self.distance_gauss_occupied()

        dist_free = self.distance_gauss_free()


        Woc = dist_occupied

        Wfc = np.max(2 * np.std(dist_occupied) - dist_occupied, 0)
        Wfc = dist_free

        # for each cell separately, then weigh by number of items in cell

        particle_counts = self.particle_counts()

        Noc = particle_counts
        Nfc = np.maximum(0, self.params.max_cell_particles - Noc)

        with np.errstate(divide='ignore', invalid='ignore'):
            Poc = Woc * Noc / (Woc * Noc + Wfc * Nfc)

            Pfc = Wfc * Nfc / (Woc * Noc + Wfc * Nfc) # not needed



        # resampled number of particles:

        Nrc = Poc * self.params.max_cell_particles

        res_ratio = Nrc / np.maximum(1.0, Noc) * 0.8  # this should probably be less to allow for creation of new particles


        res_ratio = np.maximum(res_ratio, self.params.min_particle_survival_probability)

        #plt.imshow(res_ratio, cmap='hot', interpolation='nearest', vmin=0.0, origin='lower')


        self._cell_weights = res_ratio

        return self._cell_weights


    def calc_particle_weights(self):

        p_x = self._particles[:, 0].astype(np.uintp)
        p_y = self._particles[:, 1].astype(np.uintp)

        p_col = self._grid.get_col_i(self._particles[:, [self.P_X, self.P_Y, self.P_Z]])
        p_row = self._grid.get_row_i(self._particles[:, [self.P_X, self.P_Y, self.P_Z]])

        res_ratio = self._cell_weights

        weights = res_ratio[p_row, p_col] # for each particle

        self._particles[:, self.P_W] = weights

        return weights

    def resample(self):


        if self._particles.size:

            self.calc_cell_weights()

            weights = self.calc_particle_weights()

            frac, inte = np.modf(weights)

            rep_cnts = inte.astype(np.int64) + (np.random.sample(size=weights.shape) < frac)

            self._particles = np.repeat(self._particles, rep_cnts, axis=0)
            self._particle_counts = None


    def sample_new(self):
        occupied_cells = self._grid.get_occupied_bin().astype(np.bool)

        grid_cols = self._grid.cols()
        grid_rows = self._grid.rows()

        rows, cols = np.meshgrid(np.arange(grid_rows, dtype=np.int), np.arange(grid_cols, dtype=np.int), indexing='ij')
        rows = rows.flatten()
        cols = cols.flatten()

        particle_counts = self.particle_counts()

        free_space = np.maximum(0, self.params.max_cell_particles - particle_counts)

        create_cnts = free_space * self.params.new_particle_ratio

        occ_rows = rows[occupied_cells[rows.astype(np.intp), cols.astype(np.intp)]]
        occ_cols = cols[occupied_cells[rows.astype(np.intp), cols.astype(np.intp)]]

        spawn_rows = (occ_rows.astype(np.float) + 0.5)
        spawn_cols = (occ_cols.astype(np.float) + 0.5)

        particle_coords = self._grid.transform_to_world(np.column_stack((self._grid.get_pos(spawn_rows, spawn_cols), np.zeros(spawn_cols.shape, dtype=spawn_cols.dtype))))

        particle_coords = np.repeat(particle_coords, np.floor(create_cnts[occ_rows, occ_cols]).astype(np.int), axis=0)

        rho = np.random.uniform(0.0, self.params.max_particle_velocity, size=(particle_coords.shape[0], 1)).astype(dtype=np.float64)
        phi = np.random.uniform(0.0, 2 * math.pi, size=(particle_coords.shape[0], 1)).astype(dtype=np.float64)

        v_x = rho * np.cos(phi)
        v_y = rho * np.sin(phi)
        v_z = np.zeros(v_x.shape, dtype=v_x.dtype)

        #add static particles

        static_particles = np.random.random(size=v_x.shape) < self.params.static_particle_ratio
        v_x[static_particles] = 0.0
        v_y[static_particles] = 0.0
        v_z[static_particles] = 0.0

        weights = np.ones(shape=v_x.shape, dtype=np.float64)
        age = np.ones(shape=v_x.shape, dtype=np.float64)

        result = np.concatenate((particle_coords, v_x, v_y, v_z, weights, age), axis=1)
        return result

    def print_summary(self):

        cnt = self._particles.shape[0]

        print "Particle count:", cnt

        grid_coords = (self._particles[:,[self.P_X, self.P_Y, self.P_Z]])

        print "X range: [{0}, {1}]".format(grid_coords[:,self.P_X].min(), grid_coords[:,self.P_X].max())
        print "Y range: [{0}, {1}]".format(grid_coords[:,self.P_Y].min(), grid_coords[:,self.P_Y].max())
        print "Z range: [{0}, {1}]".format(grid_coords[:,self.P_Z].min(), grid_coords[:,self.P_Z].max())

        particle_counts = self.particle_counts()
        print "Max particles per cell setting", self.params.max_cell_particles

        print "Min per cell:", np.min(particle_counts)
        print "Max per cell:", np.max(particle_counts)
        print "Mean per cell:", np.mean(particle_counts)
        print "Median per cell:", np.median(particle_counts)
        print ""

        if self._particles.size > 0:
            weights = self._particles[:, self.P_W]
            ages = self._particles[:, self.P_A]

            print "Min weight: ", weights.min()
            print "Max weight: ", weights.max()
            print "Mean weight: ", np.mean(weights)
            print "Median weight: ", np.median(weights)
            print ""
            print "Min age: ", ages.min()
            print "Max age: ", ages.max()
            print "Mean age: ", np.mean(ages)
            print "Median age: ", np.median(ages)

            print ""
            print "Average position and motion"
            print np.average(self._particles[:, [self.P_X, self.P_Y, self.P_Z, self.P_DX, self.P_DY, self.P_DZ]], axis=0, weights = self._particles[:, self.P_A])
            print np.std(self._particles[:, [self.P_X, self.P_Y, self.P_Z, self.P_DX, self.P_DY, self.P_DZ]], axis=0)

    def get_particles_xyza(self):

        orientation = np.arctan2(self._particles[:, self.P_DY], self._particles[:, self.P_DX]).reshape((-1, 1))

        orientation = np.where(orientation >0, orientation, 2.0 * np.pi + orientation)

        return np.hstack((self._particles[:, [self.P_X, self.P_Y, self.P_Z]], orientation))

    def particles_to_pointcloud_data(self):
        pass


    def plot_particles(self, plt):
        return plt.scatter(self._particles[:,0], self._particles[:,1], s=4, c=self._particles[:, 5], cmap='viridis', marker=',', linewidths=0, edgecolor=None, vmin=1, vmax=10)

    def update_plot(self, sc):
        sc.set_offsets(self._particles[:,0:2])
        sc.set_array(self._particles[:, 5])


    def plot_cell_weights(self, plt):
        return plt.imshow(self._cell_weights, cmap='gray', interpolation='nearest', vmin=0.0, origin='lower')

    def update_cell_weights(self, sc, plt):

        #sc.set_data(self._cell_weights)
        if np.max(self._cell_weights) > 0:
            plt.imshow(self._cell_weights, cmap='gray', interpolation='nearest', vmin=0.0, origin='lower')


    def plot_particle_density(self, plt):
        grid_conf = self._grid.get_config()

        particle_counts = self.particle_counts()

        return plt.imshow(particle_counts, cmap='gray', interpolation='nearest', vmin=0.0, origin='lower')

    def update_density(self, sc, plt):
        grid_conf = self._grid.get_config()

        particle_counts = self.particle_counts()
        plt.imshow(particle_counts, cmap='gray', interpolation='nearest', vmin=0.0, origin='lower')
        #sc.set_data(particle_counts) # not working for whatever reason



    def get_speeds(self):
        grid_conf = self._grid.get_config()

        rows, cols = np.meshgrid(np.arange(grid_conf.rows, dtype=np.intp), np.arange(grid_conf.cols, dtype=np.intp), indexing='ij')
        rows = rows.flatten()
        cols = cols.flatten()

        speeds = np.zeros(shape=(grid_conf.rows, grid_conf.cols))

        particle_counts = self.particle_counts()


        occupied_cells = particle_counts > 0.0
        #keep only cells tht are occupied
        occ_rows = rows[occupied_cells[rows, cols]]
        occ_cols = cols[occupied_cells[rows, cols]]

        for i in xrange(occ_rows.shape[0]):
            mask = np.logical_and(self._particles[:, 0].astype(np.intp) == occ_cols[i], self._particles[:, 1].astype(np.intp) == occ_rows[i])#, self._particles[:, 5] >=2 )
            cell_particles = self._particles[mask, :]
            if cell_particles.size > 0:
                #try:
                    age_weight = 1 + np.maximum(0.0, np.log(cell_particles[:, 5]))
                    # cell_particles[:, 4] *
                    avg = np.average(cell_particles[:, 2:4], axis=0, weights=age_weight).reshape((1, 2))
                    speeds[occ_rows[i], occ_cols[i]] = np.linalg.norm(avg)
                #except:
                #    pass

        return speeds

    def plot_speeds(self, plt):
        speeds = self.get_speeds()
        plt.imshow(speeds, cmap='gray', interpolation='nearest', vmin=0.0, origin='lower')


    def get_vectors(self):
        grid_conf = self._grid.get_config()

        rows, cols = np.meshgrid(np.arange(grid_conf.rows, dtype=np.intp), np.arange(grid_conf.cols, dtype=np.intp), indexing='ij')
        rows = rows.flatten()
        cols = cols.flatten()

        particle_counts = self.particle_counts()


        occupied_cells = particle_counts > 0.0
        #keep only cells tht are occupied
        occ_rows = rows[occupied_cells[rows, cols]]
        occ_cols = cols[occupied_cells[rows, cols]]

        vectors = np.zeros(shape=(0, 4), dtype=np.float64)

        for i in xrange(occ_rows.shape[0]):
            mask = np.logical_and(self._particles[:, 0].astype(np.intp) == occ_cols[i], self._particles[:, 1].astype(np.intp) == occ_rows[i])#, self._particles[:, 5] >=2 )
            cell_particles = self._particles[mask, :]
            if cell_particles.size > 0:
                try:
                    age_weight = np.maximum(0.0, np.log(cell_particles[:, 5]))
                    # cell_particles[:, 4] *
                    avg = np.average(cell_particles[:, 0:4], axis=0, weights=age_weight).reshape((1, 4))
                    vectors = np.append(vectors, avg, axis=0)
                except:
                    pass

        return vectors

    def plot_vectors(self, plt):

        vectors = self.get_vectors()
        print "Vectors: ", vectors.size
        return plt.quiver(vectors[:, 0], vectors[:, 1], vectors[:, 2], vectors[:, 3], color=(0.0, 1.0, 0.0, 1.0), scale=1.0, units='xy', width=0.2)


    def update_vectors(self, quiver):

        vectors = self.get_vectors()
        print "Vectors: ", vectors.size
        quiver.set_offsets(vectors[:, 0:2])
        quiver.set_UVC(vectors[:, 2] * 2, vectors[:, 3] * 2)






