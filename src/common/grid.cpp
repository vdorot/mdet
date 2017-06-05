//
// Created by viktor on 12.3.17.
//

#include "ros/ros.h"
#include "std_msgs/String.h"
#include "geometry_msgs/Pose.h"
#include "nav_msgs/OccupancyGrid.h"
#include "std_msgs/Header.h"
#include <tf/transform_broadcaster.h>
#include <sstream>
#include <tf/transform_listener.h>

#include "tf2_geometry_msgs/tf2_geometry_msgs.h"
#include "tf2_eigen/tf2_eigen.h"
#include "tf2_sensor_msgs/tf2_sensor_msgs.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sensor_msgs/LaserScan.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>

using namespace std;
using namespace cv;

vector<int8_t> matToOccupancyGrid(Mat cartMap){

    Mat flippedMap;
    flip(cartMap, flippedMap, -1); // flip around both axes

    vector<int8_t> result;
    if (flippedMap.isContinuous()) {
        result.assign(flippedMap.datastart, flippedMap.dataend);
    } else {
        for (int i = 0; i < flippedMap.rows; ++i) {
            result.insert(result.end(), flippedMap.ptr<uchar>(i), flippedMap.ptr<uchar>(i)+flippedMap.cols);
        }
    }
    return result;
}

Mat occupancyGridToMat(nav_msgs::OccupancyGrid::ConstPtr occupancyGrid){

    //https://github.com/tu-darmstadt-ros-pkg/hector_slam/blob/catkin/hector_compressed_map_transport/src/map_to_image_node.cpp#L109

    int size_x = occupancyGrid->info.width;
    int size_y = occupancyGrid->info.height;

    const std::vector<int8_t> map_data (occupancyGrid->data);


    Mat map = cv::Mat(size_y, size_x, CV_8U);

    unsigned char *map_mat_data_p = map.data;

    for (int y = 0; y < size_y; ++y){

        int idx_map_y = size_x * y;
        int idx_img_y = size_x * y;

        for (int x = 0; x < size_x; ++x){

            int idx = idx_img_y + x;

            map_mat_data_p[idx] = map_data[idx_map_y + x];

        }
    }

    Mat flippedMap = cv::Mat(size_y, size_x, CV_8U);
    flip(map, flippedMap, -1);

    return flippedMap;
}

void publishOccupancyGrid(ros::Publisher publisher, Mat cartMap, std_msgs::Header header, float resolution){

    nav_msgs::OccupancyGrid localGrid ;
    localGrid.header = header;
    localGrid.info.map_load_time = header.stamp;
    localGrid.info.resolution = resolution;
    localGrid.info.width = cartMap.cols;
    localGrid.info.height = cartMap.rows;
    localGrid.info.origin.position = geometry_msgs::Point();
    localGrid.info.origin.position.x -= (cartMap.cols * resolution) / 2.0;
    localGrid.info.origin.position.y -= (cartMap.rows * resolution) / 2.0;
    localGrid.info.origin.orientation = tf2::toMsg(tf2::Quaternion::getIdentity());
    localGrid.data = matToOccupancyGrid(cartMap);
    publisher.publish(localGrid);
}