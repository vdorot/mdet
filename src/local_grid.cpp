
#include "common/options.h"
#include "ros/ros.h"
#include "std_msgs/String.h"
#include "geometry_msgs/Pose.h"
#include "nav_msgs/OccupancyGrid.h"
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>

#include "tf2_geometry_msgs/tf2_geometry_msgs.h"
#include "tf2_eigen/tf2_eigen.h"
#include "tf2_sensor_msgs/tf2_sensor_msgs.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sensor_msgs/LaserScan.h>

#include "common/grid.h"

using namespace cv;
using namespace std;



string TARGET_FRAME;

float ANGLE_INCREMENT;
float RANGE_INCREMENT;

bool LASER_SCAN_ASSUME_FREE;
float MAX_RANGE;


ros::Publisher localGridPublisher;
ros::Publisher localGridInversePublisher;

double polarRadius(double x, double y) {
    return hypot(x, y);
}

float polarPhi(float x, float y){
    return atan2(y, x);
}

//assuming sensor is at origin, z pointing up
//assuming free space where no measurements
void addPointCloudMeasurement(Mat polarMap, float phi, float r, float angleIncrement, float rangeIncrement){
    int row = floor((phi + M_PI) / angleIncrement);
    int col = floor(r / rangeIncrement);
    if((row >=0 && row < polarMap.rows) && (col >=0 && col < polarMap.cols)) {
        polarMap.at<uint8_t>(row, col) = OCCUPANCY_GRID_CELL_OCCUPIED;
    }
}


Mat pointcloudToPolarMap(sensor_msgs::PointCloud2 msg){

    int rangeBins = MAX_RANGE / RANGE_INCREMENT;

    int rays = 2 * M_PI / ANGLE_INCREMENT;

    Mat polarMap(rays, rangeBins, CV_8U);
    polarMap.setTo(OCCUPANCY_GRID_CELL_FREE);


    sensor_msgs::PointCloud2Iterator<float> iter_x(msg, "x");

    for (; iter_x != iter_x.end(); ++iter_x) {
        float phi = polarPhi(iter_x[0], iter_x[1]);
        float r = polarRadius(iter_x[0], iter_x[1]);
        addPointCloudMeasurement(polarMap, phi, r, ANGLE_INCREMENT, RANGE_INCREMENT);
    }

    return polarMap;
}



void addLaserScanMeasurement(Mat polarMap, float phi, float r, float angleIncrement, float rangeIncrement, bool assumeFree){

    int row = floor((phi + M_PI) / angleIncrement);
    int col = floor(r / rangeIncrement);

    if(row >= 0 && row < polarMap.rows && col >=0){

        if(!assumeFree){
            for(int i=0; i< min(polarMap.cols, col); i++){
                if(i < col){
                    polarMap.at<uint8_t>(row,i) = (polarMap.at<uint8_t>(row,i) == OCCUANCY_GRID_CELL_UNKNOWN) ? OCCUPANCY_GRID_CELL_FREE : polarMap.at<uint8_t>(row,i);
                } else {
                    polarMap.at<uint8_t>(row,i) = OCCUPANCY_GRID_CELL_OCCUPIED;
                }
            }
        }else {

            for(int i=0; i< polarMap.cols; i++){
                if(i < col){
                    // do nothing
                } else if(i == col){
                    polarMap.at<uint8_t>(row,i) = OCCUPANCY_GRID_CELL_OCCUPIED;
                } else {
                    polarMap.at<uint8_t>(row,i) = (polarMap.at<uint8_t>(row,i) != OCCUPANCY_GRID_CELL_OCCUPIED) ? OCCUANCY_GRID_CELL_UNKNOWN : polarMap.at<uint8_t>(row,i);
                }
            }
        }

    }
}



Mat laserScanToPolarMap(sensor_msgs::LaserScan msg){
    int rangeBins = MAX_RANGE / RANGE_INCREMENT;

    int rays = 2*M_PI / msg.angle_increment;

    Mat polarMap(rays, rangeBins, CV_8U);
    if(LASER_SCAN_ASSUME_FREE) {
        polarMap.setTo(OCCUPANCY_GRID_CELL_FREE);
    }else{
        polarMap.setTo(OCCUANCY_GRID_CELL_UNKNOWN);
    }
    for(int i=0; i < msg.ranges.size(); i++){
        float angle = msg.angle_min + i * msg.angle_increment;
        float range = msg.ranges[i];
        if (range > msg.range_min && range < msg.range_max) {
            addLaserScanMeasurement(polarMap, angle, range, msg.angle_increment, RANGE_INCREMENT, LASER_SCAN_ASSUME_FREE);
        }
    }
    return polarMap;
}

Mat convertPolarToCart(Mat polarMap){

    int radius = MAX_RANGE / RANGE_INCREMENT;

    Mat result(radius, radius, CV_8U);

    Mat resampledPolar(radius, radius, CV_8U);
    resize(polarMap, resampledPolar, Size(radius, radius), 0, 0, INTER_LINEAR);

    Point2f center( result.cols / 2.0, result.rows / 2.0 );

    linearPolar(resampledPolar, result, center, radius, WARP_INVERSE_MAP + INTER_LINEAR);
    return result;
}

void laserScanCallback(const sensor_msgs::LaserScan::ConstPtr& msg)
{


    Mat polarMap = laserScanToPolarMap(*msg);

    Mat cartMap = convertPolarToCart(polarMap);

    publishOccupancyGrid(localGridPublisher, cartMap, msg->header, RANGE_INCREMENT);
}

tf2_ros::Buffer *tfBuffer;
tf2_ros::TransformListener *tfListener;

void pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg)
{

    ROS_INFO("Local_grid: point cloud received");

    try {

        sensor_msgs::PointCloud2 transformedPointCloud;

        tfBuffer->transform(*msg, transformedPointCloud, TARGET_FRAME, ros::Duration(1.0));
        transformedPointCloud.header.frame_id = TARGET_FRAME;
        Mat polarMap = pointcloudToPolarMap(transformedPointCloud);

        Mat cartMap = convertPolarToCart(polarMap);

        publishOccupancyGrid(localGridPublisher, cartMap, transformedPointCloud.header, RANGE_INCREMENT);
        publishOccupancyGrid(localGridInversePublisher, 255 - cartMap, transformedPointCloud.header, RANGE_INCREMENT);

    }catch(tf2::TransformException &ex){
        ROS_WARN("Could NOT transform %s", ex.what());
    }


}

int main(int argc, char **argv)
{

    ros::init(argc, argv, "local_grid");


    ros::NodeHandle n;

    ros::param::param<std::string>("~target_frame", TARGET_FRAME, "odom");

    ros::param::param<float>("~angle_increment", ANGLE_INCREMENT, 2 * M_PI / (4500 / 4));  // 4500 - velodyne 0.08deg increment, /4 to smear small objects

    ros::param::param<float>("~max_range", MAX_RANGE, 100.0);  // meters

    ros::param::param<float>("~range_increment", RANGE_INCREMENT, 0.2);  // meters

    ros::param::param<bool>("~laser_scan_assume_free", LASER_SCAN_ASSUME_FREE, true);  // meters


    localGridPublisher = n.advertise<nav_msgs::OccupancyGrid>("local_grid", 1);
    localGridInversePublisher = n.advertise<nav_msgs::OccupancyGrid>("local_grid_inverse", 1);

    tf2_ros::Buffer tfBuf;
    tfBuffer = &tfBuf;
    tf2_ros::TransformListener tfLsn(tfBuf);
    tfListener = &tfLsn;

    ros::Subscriber scanSub = n.subscribe("laser_scan", 1, laserScanCallback);

    ros::Subscriber pointCloudSub = n.subscribe("point_cloud_scan", 1, pointCloudCallback);

    ros::spin();

    return 0;
}