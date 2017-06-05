import numpy as np
import cv2

from dst import MassFunction
import tf
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Quaternion
from std_msgs.msg import Header
from common import occupancy_grid2ndarray

class Map(object):


    def __init__(self, dst_universe):
        self.origin = None  # coords of lower left corner in real space
        self.resolution = None # cells per meter
        self.shape = None
        self.map = MassFunction(dst_universe)

    def coords_to_map(self, translation):
        world_pos = translation
        col = (world_pos[0] // self.resolution).astype(np.int32)
        row = (world_pos[1] // self.resolution).astype(np.int32)
        return np.array([row, col], dtype=np.int32)

    def map_range(self, translation, shape, center=None):

        if center is not None:
            translation -= center * self.resolution

        map_pos = self.coords_to_map(translation - self.origin)

        return map_pos, map_pos + shape

    def calc_world_range(self, translation, shape, center=None):

        pos = translation

        if center is not None:
            pos -= center * self.resolution

        shape = shape[[1, 0]].astype(np.float32) * self.resolution

        return pos, pos + shape

    def world_range(self):
        return self.calc_world_range(self.origin, self.shape)

    def enlarge_map(self, translation, center, shape, dtype, focal_sets=None):

        prev_origin = self.origin
        prev_shape = self.shape
        if self.origin is None or self.shape is None:

            new_map_pos = self.coords_to_map(translation) - center

            new_map_end = new_map_pos + shape

        else:
            cur_map_pos = self.coords_to_map(self.origin)

            cur_map_end = cur_map_pos + self.shape

            update_map_pos = self.coords_to_map(translation) - center

            update_map_end = update_map_pos + shape

            new_map_pos = np.minimum(cur_map_pos, update_map_pos)

            new_map_end = np.maximum(cur_map_end, update_map_end)

        self.origin = (new_map_pos * self.resolution)[[1, 0]]

        self.shape = new_map_end - new_map_pos

        if focal_sets is None:
            focal_sets = self.map.get_focal_sets(include_universe=False)

        for s in focal_sets:
            # for each focal in self.map, recreate larger matrix and fill existing data
            prev_map = self.map[s]
            self.map[s] = np.zeros(shape=self.shape, dtype=dtype)

            if prev_origin is not None and type(prev_map) is np.ndarray:
                region_s, region_e = self.map_range(prev_origin, prev_shape)
                self.map[s][region_s[0]: region_e[0], region_s[1]:region_e[1]] = prev_map

        self.map.normalize_universe()


    def rotate_bound(self, image, angle):
        # grab the dimensions of the image and then determine the
        # center
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)

        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), -(angle / (2.0 * np.pi) * 360.0), 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        # perform the actual rotation and return the image
        return cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0), nW/2, nH/2



    def update_map(self, grid, m):
        """
        Update map
        :param OccupancyGrid grid:
        :param MassFunction m:
        :return:
        """

        if self.shape is None:
            self.resolution = grid.resolution()
        else:
            if grid.resolution() != self.resolution:
                raise ValueError("Incompatible grid")

        translation = grid.transform_to_world(np.array([grid.width() / 2, grid.height() / 2, 0.0], dtype=np.float))[0, [0,1]] # middle of grid

        rotation_angle = grid.get_rotation_angle()

        rot_c = np.array([0, 0], dtype=np.int32)  # rotation center, rows, cols
        rot_shape = np.array([0, 0], dtype=np.int32)  # rotated image size

        for set in m.get_focal_sets(include_universe=False):
            m[set], rot_c[1], rot_c[0] = self.rotate_bound(m[set], rotation_angle)
            rot_shape = np.array(m[set].shape, dtype=np.int32)

        self.enlarge_map(translation, rot_c, rot_shape, np.float32, focal_sets=m.get_focal_sets(include_universe=False))

        start, end = self.map_range(translation, rot_shape, rot_c)

        print("Updating map at range: {0} to {1}".format(str(tuple(start)), str(tuple(end))))

        region_m = MassFunction(self.map.get_universe())

        for s in self.map.get_focal_sets(include_universe=False):
            region_m[s] = self.map[s][start[0]:end[0], start[1]:end[1]]

        region_m.normalize_universe()

        region_m.josang_cumulative(m)

        for s in self.map.get_focal_sets(include_universe=False):
            self.map[s][start[0]:end[0], start[1]:end[1]] = region_m[s]


    def get_map(self):
        return self.map

    def publish_map(self, publisher, mass, frame_id, stamp):
        grid_msg = OccupancyGrid()
        grid_msg.header = Header()
        grid_msg.header.frame_id = frame_id
        grid_msg.header.stamp = stamp
        grid_msg.info.map_load_time = stamp
        grid_msg.info.resolution = self.resolution

        grid_msg.info.width = self.shape[1]
        grid_msg.info.height = self.shape[0]

        grid_msg.info.origin.position.x = self.origin[0]
        grid_msg.info.origin.position.y = self.origin[1]
        grid_msg.info.origin.orientation = Quaternion(*tf.transformations.quaternion_from_euler(0.0, 0.0, 0.0))

        grid_msg.data = (mass * 255).astype(np.int8).ravel().tolist()
        publisher.publish(grid_msg)

    @classmethod
    def map_from_msg(cls, grid_msg, dst_universe, item):

        map = cls(dst_universe)
        map.origin = np.array([grid_msg.info.origin.position.x, grid_msg.info.origin.position.y], dtype=np.float32)
        map.resolution = grid_msg.info.resolution
        map.shape = np.array([grid_msg.info.width, grid_msg.info.height], dtype=np.float32)
        map.map[item] = occupancy_grid2ndarray(grid_msg)

        return map



