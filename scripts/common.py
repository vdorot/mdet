import tf
from std_msgs.msg import Header
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Quaternion
from sensor_msgs.msg import PointCloud2, PointField
import numpy as np


def publish_grid(publisher, grid_array, msg_header, resolution):
    """
    Publish numpy grid

    :param rospy.Publisher publisher:
    :param np.ndarray grid_array:
    :param Header msg_header:
    :param float resolution:
    :return:
    """
    grid_msg = OccupancyGrid()
    grid_msg.header = msg_header
    grid_msg.info.map_load_time = msg_header.stamp
    grid_msg.info.resolution = resolution
    rows = grid_array.shape[0]
    cols = grid_array.shape[1]

    grid_msg.info.width = cols
    grid_msg.info.height = rows

    grid_msg.info.origin.position.x -= (cols * resolution) / 2.0
    grid_msg.info.origin.position.y -= (rows * resolution) / 2.0
    grid_msg.info.origin.orientation = Quaternion(*tf.transformations.quaternion_from_euler(0.0, 0.0, 0.0))

    grid_msg.data = ndarray2occupancy_grid_data(grid_array)
    publisher.publish(grid_msg)


def ndarray2occupancy_grid_data(grid_array):
    """
    :param np.ndarray grid_array:
    :return: Occupancy grid data
    :rtype: np.ndarrary, list
    """
    return grid_array.ravel().tolist()


def occupancy_grid2ndarray(occupancy_grid):
    """
    Convert ROS OccupancyGrid to numpy
    :param OccupancyGrid occupancy_grid:
    :return: Converted grid
    :rtype: np.ndarray
    """
    cols = occupancy_grid.info.width
    rows = occupancy_grid.info.height
    return np.asarray(occupancy_grid.data, dtype=np.uint8).reshape(rows, cols)


def survival_map_to_grid(survival_map):
    return (survival_map * 255).astype(np.int8)


def publish_point_cloud(particle_publisher, cloud_points, frame_id, stamp):

    header = Header()
    header.stamp = stamp
    header.frame_id = frame_id

    fields = [PointField('x', 0, PointField.FLOAT32, 1),
              PointField('y', 4, PointField.FLOAT32, 1),
              PointField('z', 8, PointField.FLOAT32, 1)]

    if cloud_points.shape[1] == 4:
        fields.append(PointField('intensity', 8, PointField.FLOAT32, 1))

    buff = np.ascontiguousarray(cloud_points, dtype=np.float32).ravel().tobytes()

    cloud = PointCloud2(header=header,
                        height=1,
                        width=cloud_points.shape[0],
                        is_dense=False,
                        is_bigendian=False,
                        fields=fields,
                        point_step=len(fields)*4,
                        row_step=len(fields) * 4 * cloud_points.shape[1],
                        data=buff)

    particle_publisher.publish(cloud)
