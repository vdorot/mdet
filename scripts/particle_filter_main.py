#!/usr/bin/env python

import rospy

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose, Point
import copy

from dst import GridMassFunction

import matplotlib.pyplot as plt
plt.ion()

from particle_filter import ParticleFilter, ParticleFilterParams
from occupancy_grid import LocalOccupancyGrid, LocalOccupancyGridParams
from tf import transformations
from tf.listener import xyz_to_mat44, xyzw_to_mat44

from common import *

from plot import ParticleFilterPlot

from mdet.msg import GridDST

import cProfile, pstats, StringIO

profiler = None


# uncomment to enable particle filter profiling

# profiler = cProfile.Profile()


transformer = None
"""
:type: tf.Transformer
"""

grid_publisher = None
dynamic_publisher = None
static_publisher = None

dynamics_plot = None
dynamics_image = None


params = ParticleFilterParams(rospy)

particle_filter = ParticleFilter(params)

particle_publisher = None
loader_publisher = None

map = None



class MyTransformer(tf.TransformListener):

    def transformPose(self, target_frame, ps, header=None):

        if hasattr(ps, "header"):
            return super(MyTransformer, self).transformPose(target_frame, ps)

        self.waitForTransform(header.frame_id, target_frame, header.stamp, rospy.Duration(4.0))

        translation, rotation = self.lookupTransformFull(target_frame=target_frame, source_frame=header.frame_id, source_time=header.stamp, target_time=header.stamp, fixed_frame='world')
        mat44 = self.fromTranslationRotation(translation, rotation)


        print("Pose: ", ps.position.x, ps.position.y, ps.position.z)

        # pose44 is the given pose as a 4x4
        pose44 = np.dot(xyz_to_mat44(ps.position), xyzw_to_mat44(ps.orientation))

        # txpose is the new pose in target_frame as a 4x4
        txpose = np.dot(mat44, pose44)

        # xyz and quat are txpose's position and orientation
        xyz = tuple(transformations.translation_from_matrix(txpose))[:3]
        quat = tuple(transformations.quaternion_from_matrix(txpose))

        # assemble return value PoseStamped
        return Pose(Point(*xyz), Quaternion(*quat))

scan_cnt = 0

def grid_callback(input_grid):
    """
    :param OccupancyGrid input_grid:
    """
    global grid_publisher
    global transformer
    global particle_publisher
    global loader_publisher
    global grid_publisher
    global dynamic_publisher
    global static_publisher
    global scan_cnt
    global tfbr
    global plot
    global map

    global hist_plot
    global local_environment_publisher
    cur_scan_cnt = scan_cnt
    scan_cnt += 1
    cur_scan_cnt +=1

    print("Got scan request {0}".format(cur_scan_cnt))

    try:
        # transform grid pose to world tf

        target_frame = "world"

        target_time = input_grid.header.stamp

        #translation, rotation = transformer.lookupTransformFull(target_frame=target_frame, source_frame=input_grid.header.frame_id, source_time=target_time, target_time=target_time, fixed_frame='world')

        input_grid.info.origin = transformer.transformPose(target_frame, input_grid.info.origin, input_grid.header)
        input_grid.header.frame_id = target_frame

        print("tx pose:", input_grid.info.origin.position.x, input_grid.info.origin.position.y, input_grid.info.origin.position.z)

        params = LocalOccupancyGridParams(rospy, '~grid_')

        grid = LocalOccupancyGrid(input_grid, params)

        # inline profiling, limited to particle filter
        if profiler:
            profiler.enable()
        particle_filter.update_grid(grid)
        if profiler:
            profiler.disable()


        distance = (particle_filter.distance_gauss_occupied() * 255).astype(np.int8)
        input_grid.header.frame_id = "odom"
        publish_grid(grid_publisher, distance, input_grid.header, input_grid.info.resolution)

        if profiler:
            profiler.enable()
        particle_filter.step_model()
        if profiler:
            profiler.disable()


        mass_static, mass_dynamic = particle_filter.get_masses()

        universe = ['F', 'D', 'S']

        m_grid = GridMassFunction(universe, grid.info)
        m_grid[['S', 'D']] = grid.get_grid()
        m_grid['F'] = 1.0 - m_grid[['S', 'D']]

        m_pfilter = GridMassFunction(universe, grid.info)
        m_pfilter['S'] = mass_static
        m_pfilter['D'] = mass_dynamic

        m_pfilter.normalize_universe()

        # m_pfilter[universe] is automatically set to the remainder

        m, conflict = m_grid.conj_dempster(m_pfilter)

        m['F'] += conflict

        mass_static = m['S']

        mass_dynamic = m['D']

        #  publishing of dynamic and static masses

        print("Publishing state")

        static_scaled = ((1.0 - mass_static) * 255).astype(np.int8)

        publish_grid(static_publisher, static_scaled, input_grid.header, input_grid.info.resolution)

        dynamic_scaled = ((1.0 - mass_dynamic) * 255).astype(np.int8)

        publish_grid(dynamic_publisher, dynamic_scaled, input_grid.header, input_grid.info.resolution)


        local_environment_publisher.publish(m.asMessage(input_grid.header))

        orientation = particle_filter.get_orientation_means()

        plot.update_plot(mass_static, mass_dynamic, orientation, grid.get_rotation_angle())


        if profiler:
            profiler.enable()
        particle_filter.resample()
        if profiler:
            profiler.disable()


        if profiler:
            s = StringIO.StringIO()
            sortby = 'cumulative'
            ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
            ps.print_stats()

            file = "/home/viktor/Sync/School/DP/profile.txt"

            with open(file, "w") as f:
                f.write(s.getvalue())



        publish_point_cloud(particle_publisher, particle_filter.get_particles_xyza(), target_frame, input_grid.header.stamp)

    finally:
        #time.sleep(1.6)
        # print("Publishing scan request {0}".format(cur_scan_cnt))
        # loader_publisher.publish(Empty())
        pass









def mapping():

    # get mass function after fusion of grid and particle filter

    # get 2d rotation matrix from grid pose
    # get translation from grid pose

    # for all non-zero masses in mass function, use opencv warpAffine to rotate grid to correct position
    # calculate affected region fo the map
    # enlarge map if necessary
    # create new MassFunction with only affected region of map
    # join rotated mass function to region of map using Josang cumulative operator



    # http://stackoverflow.com/questions/22041699/rotate-an-image-without-cropping-in-opencv-in-c

    pass




def main():
    global grid_publisher
    global dynamic_publisher
    global static_publisher
    global particle_filter
    global transformer
    global particle_publisher
    global loader_publisher
    global dynamics_plot
    global plot
    global local_environment_publisher
    rospy.init_node('particle_filter')


    transformer = MyTransformer(False)

    rospy.Subscriber("local_grid", OccupancyGrid, grid_callback)

    grid_publisher = rospy.Publisher('distance_transform', OccupancyGrid, queue_size=1)

    dynamic_publisher = rospy.Publisher('dynamic', OccupancyGrid, queue_size=1)

    static_publisher = rospy.Publisher('static', OccupancyGrid, queue_size=1)

    particle_publisher = rospy.Publisher('particles', PointCloud2, queue_size=1)

    local_environment_publisher = rospy.Publisher('local_environment', GridDST, queue_size=1)

    plot = ParticleFilterPlot()

    rospy.spin()


if __name__ == '__main__':
    try:
        print("Particle filter starting")
        main()
    except rospy.ROSInterruptException:
        print("Interrupted by ROS")
        pass