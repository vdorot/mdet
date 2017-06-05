//
// Created by viktor on 12.3.17.
//

#ifndef PROJECT_UTILS_H
#define PROJECT_UTILS_H

#include "options.h"
#include <opencv2/core/core.hpp>



using namespace std;
using namespace cv;

const uint8_t OCCUPANCY_GRID_CELL_FREE = 0;
const uint8_t OCCUPANCY_GRID_CELL_OCCUPIED = 255;
const uint8_t OCCUANCY_GRID_CELL_UNKNOWN = 127;

vector<int8_t> matToOccupancyGrid(Mat cartMap);

Mat occupancyGridToMat(nav_msgs::OccupancyGrid::ConstPtr occupancyGrid);

//resolution: meters per grid cell
void publishOccupancyGrid(ros::Publisher publisher, Mat cartMap, std_msgs::Header header, float resolution);

#endif //PROJECT_UTILS_H
