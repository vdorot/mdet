#!/usr/bin/env python2
from __future__ import print_function
import sys
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

STATIC_THRESHOLD = 0.08


class PointCloudFilterNode(object):

    def __init__(self):

        rospy.init_node('pcl_filter')

        self.tf_listener = TransformListener()

        rospy.Subscriber("point_cloud_obstacles", PointCloud2, lambda point_cloud: self.obstacle_cloud_handler(point_cloud))

        rospy.Subscriber("static", OccupancyGrid, lambda grid: self.map_handler(grid))

        self.static_point_cloud_publisher = rospy.Publisher('point_cloud_static', PointCloud2, queue_size=1)

        self.dynamic_point_cloud_publisher = rospy.Publisher('point_cloud_dynamic', PointCloud2, queue_size=1)

        self.map = None

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

        if self.map is None:
            print("Cannot filter point cloud, no map received yet", file=sys.stderr)
            return

        point_cloud = self.point_cloud_to_np_array(point_cloud_msg)
        point_cloud = self.transform_cloud(point_cloud, point_cloud_msg.header, self.map.header.frame_id)

        grid = LocalOccupancyGrid(self.map, LocalOccupancyGridParams(rospy, '~grid_'))

        cols = grid.get_col_i(point_cloud[:, 0:3])
        rows = grid.get_row_i(point_cloud[:, 0:3])

        valid_pos = np.logical_and(
            np.logical_and(cols >= 0, cols < grid.cols()),
            np.logical_and(rows >= 0, rows < grid.rows()),
        )

        obstacle_cloud = point_cloud[valid_pos]

        cols = grid.get_col_i(obstacle_cloud[:, 0:3])
        rows = grid.get_row_i(obstacle_cloud[:, 0:3])

        threshold = STATIC_THRESHOLD

        obstacle_static_cloud = obstacle_cloud[(1 - grid.get_grid())[rows, cols] > threshold, :]

        obstacle_dynamic_cloud = obstacle_cloud[(1 - grid.get_grid())[rows, cols] < threshold, :]

        now = rospy.Time.now()

        print("Publishing filtered point clouds, ({},{})".format(obstacle_static_cloud.shape[0], obstacle_dynamic_cloud.shape[0]))

        publish_point_cloud(self.static_point_cloud_publisher, obstacle_static_cloud, self.map.header.frame_id, now)
        publish_point_cloud(self.dynamic_point_cloud_publisher, obstacle_dynamic_cloud, self.map.header.frame_id, now)


    def map_handler(self, map):

        self.map = map

        print("Static map received")


if __name__ == '__main__':
    try:
        PointCloudFilterNode()
    except rospy.ROSInterruptException:
        print("Interrupted by ROS")
        pass