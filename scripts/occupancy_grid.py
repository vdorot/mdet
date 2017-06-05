from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose
import tf
import tf_conversions

import tf2_ros
import tf2_geometry_msgs


import transforms3d, transforms3d.euler
import numpy as np

from params import RosParams

class LocalOccupancyGridParams(RosParams):

    _params = [('occupancy_threshold', 0.7, float, 'occupancy_threshold')]


class LocalOccupancyGrid:


    DTRANSFORM_OCCUPANCY_THRESHOLD = 0.6


    def _ndarray2occupancy_grid_data(grid_array):
        """
        :param np.ndarray grid_array:
        :return: Occupancy grid data
        :rtype: np.ndarrary, list
        """
        return grid_array.ravel().tolist()


    def _occupancy_grid2ndarray(self, occupancy_grid):
        """
        Convert ROS OccupancyGrid to numpy
        :param OccupancyGrid occupancy_grid:
        :return: Converted grid
        :rtype: np.ndarray
        """
        cols = occupancy_grid.info.width
        rows = occupancy_grid.info.height
        return np.asarray(occupancy_grid.data, dtype=np.uint8).reshape(rows, cols) / 255.0


    def __init__(self, occupancy_grid=None, params=LocalOccupancyGridParams()):
        """

        :param OccupancyGrid occupancy_grid:
        """
        if occupancy_grid is not None:
            self.info = occupancy_grid.info
            self.grid_mat = self._occupancy_grid2ndarray(occupancy_grid)
        self.params = params

    @classmethod
    def fromMetadata(cls, metadata, params=LocalOccupancyGridParams()):
        grid = cls(params=params)
        grid.info = metadata
        return grid

        # data is empty, Grid is just used for coordinate transformation logic in this case

    def get_grid(self):
        """
        :return: Grid as Numpy array
        :rtype: np.ndarray
        """
        return self.grid_mat

    def width(self):
        return self.cols() * self.resolution()

    def height(self):
        return self.rows() * self.resolution()

    def resolution(self):
        return self.info.resolution

    def cols(self):
        return self.info.width

    def rows(self):
        return self.info.height

    def _pose_to_rot(self, pose):
        """

        :param Pose pose:
        :return: Rotation matrix
        :rtype: np.ndarray
        """
        rot = transforms3d.quaternions.quat2mat([pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z])

        return rot

    def _pose_to_angle(self, pose):
        angles = transforms3d.euler.quat2euler([pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z])
        return angles[2]


    def _pose_to_translation(self, pose):
        tl = np.eye(1, 3, dtype=np.float32)
        tl[0, 0] = pose.position.x
        tl[0, 1] = pose.position.y
        tl[0, 2] = pose.position.z
        return tl

    def get_translation(self):
        return (self.info.origin.position.x, self.info.origin.position.y, self.info.origin.position.z)

    def get_rotation_angle(self):
        return self._pose_to_angle(self.info.origin)

    def transform_to_grid(self, points):
        """
        Transform world points to grid - x, y, z = to grid x, y
        :param points:
        :return:
        """

        rot = self._pose_to_rot(self.info.origin)
        tl = self._pose_to_translation(self.info.origin)

        return np.dot(points - tl, rot)  # inverse = transpose

    def transform_to_world(self, points):
        """
        Transform grid points to world - grid x, y to world x, y, z
        Also works for vectors
        :param points:
        :return:
        """

        rot = self._pose_to_rot(self.info.origin)
        tl = self._pose_to_translation(self.info.origin)

        return np.dot(points, rot.T) + tl  # transforming row vectors - matrix is transposed


    def get_row(self, points):
        Y = 1
        # transform points to map coords, calculate row
        points = self.transform_to_grid(points)
        return (points[:, Y]) / (self.resolution())

    def get_col(self, points):
        X = 0
        points = self.transform_to_grid(points)
        return (points[:, X]) / (self.resolution())

    def get_pos(self, rows, cols):
        # opposite of get_row

        return np.column_stack((cols * self.resolution(), rows * self.resolution()))


    def get_row_i(self, points):
        return np.floor(self.get_row(points)).astype(np.intp)

    def get_col_i(self, points):
        return np.floor(self.get_col(points)).astype(np.intp)

    def row_range(self):
        return [0.0, self.height()]

    def col_range(self):
        return [0.0, self.width()]

    def in_range(self, points):
        points = self.transform_to_grid(points)
        X = 0
        Y = 1
        col_range = self.col_range()
        row_range = self.row_range()
        return np.logical_and(
            np.logical_and(points[:, X] >= col_range[0], points[:, X] <= col_range[1]),
            np.logical_and(points[:, Y] >= row_range[0], points[:, Y] <= row_range[1])
        )

    def get_occupied_bin(self):
        return (self.grid_mat > self.params.occupancy_threshold).astype(np.uint8)