#!/usr/bin/env python3

import rospy
import numpy as np
from tf import TransformListener
from std_msgs.msg import Empty
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import TransformStamped

from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from occupancy_grid import LocalOccupancyGridParams, LocalOccupancyGrid
from common import publish_point_cloud
import time

request_publisher = None


MAP_FRAME = '/map'


STATIC_THRESHOLD = 0.03




class NumpyAccumulator(object):

    def __init__(self, columns):
        self._capacity = 32000000
        self._columns = columns
        self._array = np.empty(shape=(self._capacity, self._columns))
        self._rows = 0

    def get_rows(self):
        return self._rows

    def get(self):
        return self._array[0:self._rows, :]

    def _grow(self, required_rows):
        if required_rows > self._capacity:
            target_capacity = self._capacity
            while target_capacity < required_rows:
                target_capacity *= 2

            old_array = self._array
            self._array = np.empty(shape=(target_capacity, self._columns))
            self._array[0: self._rows] = old_array[0: self._rows]
            self._capacity = target_capacity

    def append(self, array):
        rows = array.shape[0]
        required_rows = self._rows + rows
        self._grow(required_rows)

        self._array[self._rows: required_rows, :] = array
        self._rows = required_rows


class PointCloudAccumulatorNode(object):

    def __init__(self):

        rospy.init_node('pcl_accumulator')

        self.obstacles_accumulator = NumpyAccumulator(3)
        self.ground_accumulator = NumpyAccumulator(3)

        self.tf_listener = TransformListener()

        rospy.Subscriber("point_cloud_ground", PointCloud2, lambda point_cloud: self.ground_cloud_handler(point_cloud))
        rospy.Subscriber("point_cloud_obstacles", PointCloud2, lambda point_cloud: self.obstacle_cloud_handler(point_cloud))

        rospy.Subscriber("map_static", OccupancyGrid, lambda grid: self.static_map_handler(grid))

        self.ground_point_cloud_publisher = rospy.Publisher('acc_point_cloud_ground', PointCloud2, queue_size=1)

        self.obstacles_static_point_cloud_publisher = rospy.Publisher('acc_point_cloud_obstacles_static', PointCloud2, queue_size=1)

        self.obstacles_dynamic_point_cloud_publisher = rospy.Publisher('acc_point_cloud_obstacles_dynamic', PointCloud2, queue_size=1)

        self.static_map = None

        rospy.Subscriber("request_acc", Empty, lambda request: self.request_handler(request))

        rospy.spin()


    def point_cloud_to_np_array(self, point_cloud):

        # assuming 4 float32-s for a point

        a = np.fromstring(point_cloud.data, dtype=np.float32).reshape((-1, 4))
        a[:3] = 1.0  # use homogeneous coords for matrix transformation
        return a

    def transform_cloud(self, point_cloud, msg_header, target_frame):
        self.tf_listener.waitForTransform(msg_header.frame_id, target_frame, msg_header.stamp, rospy.Duration(4.0))
        mat44 = self.tf_listener.asMatrix(target_frame, msg_header)

        # assuming point_cloud is a Nx4 matrix

        return np.dot(point_cloud, mat44.T)


    def obstacle_cloud_handler(self, point_cloud_msg):

        point_cloud = self.point_cloud_to_np_array(point_cloud_msg)
        point_cloud = self.transform_cloud(point_cloud, point_cloud_msg.header, MAP_FRAME)

        self.obstacles_accumulator.append(point_cloud[:, 0:3])

        print("Obstacle cloud received, accumulated size: {0}".format(self.obstacles_accumulator.get_rows()))


    def ground_cloud_handler(self, point_cloud_msg):

        point_cloud = self.point_cloud_to_np_array(point_cloud_msg)
        point_cloud = self.transform_cloud(point_cloud, point_cloud_msg.header, MAP_FRAME)

        self.ground_accumulator.append(point_cloud[:, 0:3])

        print("Ground cloud received, accumulated size: {0}".format(self.ground_accumulator.get_rows()))



    def static_map_handler(self, map):

        self.static_map = map

        print("Static map received")

    def request_handler(self, request):

        if self.static_map is None:
            print("No map received")
            return
        start_time = time.time()

        grid = LocalOccupancyGrid(self.static_map, LocalOccupancyGridParams(rospy, '~grid_'))

        obstacle_cloud = self.obstacles_accumulator.get()

        ground_cloud = self.ground_accumulator.get()

        cols = grid.get_col_i(obstacle_cloud)
        rows = grid.get_row_i(obstacle_cloud)

        valid_pos = np.logical_and(
            np.logical_and(cols >= 0, cols < grid.cols()),
            np.logical_and(rows >= 0, rows < grid.rows()),
        )

        obstacle_cloud = obstacle_cloud[valid_pos]


        ground_cols = grid.get_col_i(ground_cloud)
        ground_rows = grid.get_row_i(ground_cloud)

        ground_valid_pos = np.logical_and(
            np.logical_and(ground_cols >= 0, ground_cols < grid.cols()),
            np.logical_and(ground_rows >= 0, ground_rows < grid.rows()),
        )

        ground_cloud = ground_cloud[ground_valid_pos]

        cols = cols[valid_pos]
        rows = rows[valid_pos]

        threshold = STATIC_THRESHOLD

        obstacle_static_cloud = obstacle_cloud[grid.get_grid()[rows, cols] > threshold, :]

        obstacle_dynamic_cloud = obstacle_cloud[grid.get_grid()[rows, cols] < threshold, :]

        now = rospy.Time.now()

        publish_point_cloud(self.ground_point_cloud_publisher, ground_cloud, MAP_FRAME, now)

        publish_point_cloud(self.obstacles_static_point_cloud_publisher, obstacle_static_cloud, MAP_FRAME, now)

        publish_point_cloud(self.obstacles_dynamic_point_cloud_publisher, obstacle_dynamic_cloud, MAP_FRAME, now)

        print("Point clouds published, generation took {0}s".format(time.time() - start_time))

if __name__ == '__main__':
    try:
        PointCloudAccumulatorNode()
    except rospy.ROSInterruptException:
        print("Interrupted by ROS")
        pass