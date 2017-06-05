#!/usr/bin/env python

import rospy
from nav_msgs.msg import OccupancyGrid
import copy

from dst import GridMassFunction

import matplotlib.pyplot as plt
plt.ion()

from occupancy_grid import LocalOccupancyGrid
from common import *
from map import Map

from mdet.msg import GridDST


global_map = None

static_publisher = None


def local_environment_handler(dst_msg):
    """
    :param OccupancyGrid input_grid:
    """
    global static_publisher
    global global_map

    m_loaded = GridMassFunction.fromMessage(dst_msg)

    loaded_grid = LocalOccupancyGrid.fromMetadata(dst_msg.metadata)

    global_map.update_map(loaded_grid, m_loaded)

    map_static_header = copy.deepcopy(dst_msg.header)
    map_static_header.frame_id = 'map'

    mass = global_map.get_map().belief('S')

    global_map.publish_map(map_static_publisher, mass, 'map', dst_msg.header.stamp)


def main():
    global map_static_publisher
    global global_map
    rospy.init_node('map_accumulator')

    rospy.Subscriber("local_environment", GridDST, local_environment_handler)

    map_static_publisher = rospy.Publisher('map_static', OccupancyGrid, queue_size=1)

    print("Map accumulator ready")

    universe = ['F', 'D', 'S']
    global_map = Map(universe)

    rospy.spin()


if __name__ == '__main__':
    try:
        print("Particle filter starting")
        main()
    except rospy.ROSInterruptException:
        print("Interrupted by ROS")
        pass