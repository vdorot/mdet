#!/usr/bin/env python

import rospy

from std_msgs.msg import Header, Empty, Time
from sensor_msgs.msg import PointCloud2, PointField

request_publisher = None


def particle_feedback(particles):

    global request_publisher

    print("Publishing scan request")
    request_publisher.publish(Empty())



def main():
    global request_publisher
    global particle_listener
    rospy.init_node('requester')

    particles_subscriber = rospy.Subscriber("particles", PointCloud2, particle_feedback)

    request_publisher = rospy.Publisher('request_scan', Empty, queue_size=1, latch=True)


    print("Requester started")


    while(request_publisher.get_num_connections() <= 0 or particles_subscriber.get_num_connections() <= 0):
        pass


    print("Publishing first scan request")

    request_publisher.publish(Empty())

    print("Spinning")

    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        print("Interrupted by ROS")
        pass