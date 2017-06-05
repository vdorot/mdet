#!/usr/bin/env python

import numpy as np
import rospy
import tf
from std_msgs.msg import Header, Empty, Time

from common import *

import cv2

tr_broadcaster = None
""":type : tf.TransformBroadcaster"""

def clip_range(range, size):
    return max(0, min(range[0], size)), max(0, min(range[1], size))


def draw_object(img, pos_x, pos_y, width, height):

    x_range = (pos_x - width/2, pos_x + width/2)
    y_range = (pos_y - height/2, pos_y + height/2)

    x_range = clip_range(x_range, img.shape[1])
    y_range = clip_range(y_range, img.shape[0])

    img[y_range[0]:y_range[1], x_range[0]:x_range[1]] = 1.0


def draw_obstacle(img):

    pnts = np.array([(10,20),(20,10),(90,80),(80,90)])

    cv2.fillConvexPoly(img, pnts, 1.0, cv2.LINE_AA)


def request_callback(request):
    global grid_ublisher
    global tf_broadcaster
    global frame
    global speed
    global obstacle

    now = rospy.Time.now()

    grid_array = np.zeros((100, 100), dtype=np.float32)

    if speed != 0.0:



        pos_x = 20 + np.abs(np.remainder((60 +  frame*2), 120) - 60)
        pos_y = 100 - pos_x

        draw_object(grid_array, pos_x, pos_y, 20, 20)

    else:
        draw_object(grid_array, 50, 50, 30, 30)

    if obstacle:
        draw_obstacle(grid_array)

    free_noise_level = 0.003 *0

    occ_noise_level = 0.1 *0
    grid_array = np.where(grid_array == 1.0, np.where(np.random.random(size=grid_array.shape) < occ_noise_level, 1.0 - grid_array, grid_array), np.where(np.random.random(size=grid_array.shape) < free_noise_level, 1.0 - grid_array, grid_array))


    msg_header = Header()
    msg_header.frame_id = "odom"
    msg_header.stamp = now
    msg_header.seq = frame

    publish_grid(grid_publisher, survival_map_to_grid(grid_array), msg_header, 1.0)


    tf_broadcaster.sendTransform((0., 0., 0.), tf.transformations.quaternion_from_euler(0., 0., 0.), now, "odom", "map")

    frame += 1

    print("Simulated frame {0}".format(frame))



def main():
    global grid_publisher
    global tf_broadcaster
    global frame
    global speed
    global obstacle
    node_name = "local_grid_simulator"

    rospy.init_node(node_name)

    rospy.Subscriber("request_scan", Empty, request_callback)

    grid_publisher = rospy.Publisher('local_grid', OccupancyGrid, queue_size=1)

    tf_broadcaster = tf.TransformBroadcaster()

    speed = rospy.get_param(node_name + "/speed", 1.0)
    obstacle = bool(rospy.get_param(node_name + "/obstacle", True))

    frame = 0

    rospy.spin()


if __name__ == '__main__':
    try:
        print("Simulator starting")
        main()
    except rospy.ROSInterruptException:
        print("Interrupted by ROS")
        pass