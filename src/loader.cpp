
#include "common/options.h"
#include "ros/ros.h"
#include "std_msgs/String.h"
#include "std_msgs/Empty.h"
#include <sstream>
#include <iostream>

#include <string>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/regex.hpp>
#include <but_velodyne/VelodynePointCloud.h>


#include <pcl/common/transforms.h>
#include <pcl/visualization/cloud_viewer.h>

#include <sensor_msgs/PointCloud2.h>
#include <rosgraph_msgs/Clock.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <tf/transform_broadcaster.h>
#include <boost/filesystem/convenience.hpp>

#include <Eigen/Dense>
#include <tf/transform_listener.h>
#include "tf2_geometry_msgs/tf2_geometry_msgs.h"
#include "tf2_eigen/tf2_eigen.h"

using namespace but_velodyne;
using namespace Eigen;


std::string KITTI_SEQUENCE_DIR;
std::string KITTI_SEQUENCE_ANNOT_DIR;
std::string POSES_FILE;

std::string CACHE_DIR;

bool ENABLE_CACHE;
double THROTTLE_DURATION;
bool WAIT_FOR_REQUESTS;
bool INITIAL_PUBLISH;



std::vector<double> loadAnnotation(std::string fileName){


    std::vector<double> result;
    // load point cloud


    double isGround;
    std::fstream input(fileName.c_str(), std::ios::in);

    while(input >> isGround){
        result.push_back(isGround);
    }

    input.close();

    return result;


}

std::vector<Affine3d> loadPoses(std::string fileName){
    std::vector<Eigen::Affine3d> poses;
    FILE *fp = fopen(fileName.c_str(),"r");
    if (!fp)
        return poses;
    while (!feof(fp)) {
        Affine3d P = Affine3d::Identity();

        double vals[12];

        if (fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                   &vals[ 0], &vals[ 1], &vals[ 2], &vals[ 3],
                   &vals[ 4], &vals[ 5], &vals[ 6], &vals[ 7],
                   &vals[ 8], &vals[ 9], &vals[10], &vals[11] )==12) {
            P.matrix() << vals[ 0], vals[ 1], vals[ 2], vals[ 3],
                    vals[ 4], vals[ 5], vals[ 6], vals[ 7],
                    vals[ 8], vals[ 9], vals[10], vals[11],
                         0.0,      0.0,      0.0,      1.0;
            poses.push_back(P);
        }
    }
    fclose(fp);
    return poses;
}


std::vector<std::string> listFiles(std::string sequenceDir){
    const std::string target_path = sequenceDir;
    const boost::regex my_filter( ".*\\.bin" );

    std::vector< std::string > all_matching_files;

    ROS_INFO("Loader velodyne directory: %s", target_path.c_str());

    boost::filesystem::directory_iterator end_itr; // Default ctor yields past-the-end
    for( boost::filesystem::directory_iterator i( target_path ); i != end_itr; ++i )
    {

        // Skip if not a file
        if( !boost::filesystem::is_regular_file( i->status() ) ) continue;

        boost::smatch what;

        // Skip if no match for V2:
        if( !boost::regex_match( i->path().string(), what, my_filter ) ) continue;

        // File matches, store it
        all_matching_files.push_back( i->path().string() );
    }

    std::sort(all_matching_files.begin(), all_matching_files.end());

    return all_matching_files;
}



pcl::PointCloud<pcl::PointXYZ>::Ptr loadPointCloud(std::string file){
    VelodynePointCloud cloud;

    VelodynePointCloud::fromKitti(file, cloud);

    return cloud.getXYZCloudPtr();
}


void cacheFilenames(std::string cacheDir, std::string filename, std::string *cloudFile, std::string *groundFile, std::string *notGroundFile){
    boost::filesystem::path filePath(filename);
    *cloudFile = (boost::filesystem::path(cacheDir) / (filePath.filename().string() + ".cloud.pcd")).string();
    *groundFile = (boost::filesystem::path(cacheDir) / (filePath.filename().string() + ".ground.pcd")).string();
    *notGroundFile = (boost::filesystem::path(cacheDir) / (filePath.filename().string() + ".notground.pcd")).string();
}

bool tryLoadCachedFrame(std::string cacheDir, std::string filename, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr ground, pcl::PointCloud<pcl::PointXYZ>::Ptr notGround){

    boost::filesystem::path filePath(filename);
    std::string cloudFile, groundFile, notGroundFile;
    cacheFilenames(cacheDir, filename, &cloudFile, &groundFile, &notGroundFile);

    return boost::filesystem::exists(boost::filesystem::path(cloudFile)) &&
            boost::filesystem::exists(boost::filesystem::path(groundFile)) &&
            boost::filesystem::exists(boost::filesystem::path(notGroundFile)) &&
            pcl::io::loadPCDFile<pcl::PointXYZ> (cloudFile, *cloud) != -1 &&
            pcl::io::loadPCDFile<pcl::PointXYZ> (groundFile, *ground) != -1 &&
            pcl::io::loadPCDFile<pcl::PointXYZ> (notGroundFile, *notGround) != -1;
}

void saveFrameCache(std::string cacheDir, std::string filename, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr ground, pcl::PointCloud<pcl::PointXYZ>::Ptr notGround){
    boost::filesystem::path filePath(filename);
    std::string cloudFile, groundFile, notGroundFile;
    cacheFilenames(cacheDir, filename, &cloudFile, &groundFile, &notGroundFile);

    cout << cloudFile << std::endl;

    boost::filesystem::create_directories(boost::filesystem::path(cacheDir));

    pcl::io::savePCDFileBinaryCompressed(cloudFile, *cloud);
    pcl::io::savePCDFileBinaryCompressed(groundFile, *ground);
    pcl::io::savePCDFileBinaryCompressed(notGroundFile, *notGround);
}

boost::filesystem::path getAnnotationPath(std::string annotationDir, std::string pointCloudPath){
    boost::filesystem::path path(pointCloudPath);
    return boost::filesystem::path(annotationDir) / (path.filename().string() + ".ann");
}


ros::Publisher pointCloudPublisher;
ros::Publisher obstaclesPublisher;
ros::Publisher groundPublisher;

ros::Publisher clockPublisher;

ros::Publisher posePublisher;

void publishNext();

bool isRequested = false;

ros::WallTimer throttleTimer;
bool timerTicked = false;

void throttleTimerCallback(const ros::WallTimerEvent& e){
    ROS_INFO("Loader: throttle timer timeout");

    timerTicked = true;
    if(isRequested){
        publishNext();
    }
}

ros::NodeHandle *nodeHandle;

tf::TransformListener *transformListener;

std::vector<Affine3d> poses;

bool publishScan(std::vector<std::string> scanFiles, size_t position){

    std::string scanFile = scanFiles.at(position);
    boost::filesystem::path annotPath = getAnnotationPath(KITTI_SEQUENCE_ANNOT_DIR, scanFile);
    if(boost::filesystem::exists( annotPath )){

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());

        pcl::PointCloud<pcl::PointXYZ>::Ptr ground(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PointCloud<pcl::PointXYZ>::Ptr notGround(new pcl::PointCloud<pcl::PointXYZ>());

        if(!ENABLE_CACHE || !tryLoadCachedFrame(CACHE_DIR, scanFile, cloud, ground, notGround)){

            std::vector<double> groundAnnotation = loadAnnotation(annotPath.string());

            cloud = loadPointCloud(scanFile);

            ground = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
            notGround = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());

            for (int j = 0; j < cloud->size(); j++) {
                if (groundAnnotation[j] == 1.0) {
                    ground->push_back(cloud->at(j));
                } else {
                    notGround->push_back(cloud->at(j));
                }
            }
            if(ENABLE_CACHE) {
                saveFrameCache(CACHE_DIR, scanFile, cloud, ground, notGround);
            }
        }

        AngleAxis<double> aa(M_PI / 2.0, Vector3d(1.0,0.0,0.0));

        Matrix4d coordTransform;


        coordTransform <<  0.0, 0.0, 1.0, 0.0,
                       -1.0, 0.0, 0.0, 0.0,
                        0.0,-1.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 1.0;


        Transform<double, 3, Affine> coordTransformT;

        //transformation for point clouds
        coordTransformT = coordTransform;

        //transform pose
        Affine3d poseR = Affine3d(coordTransform) * poses[position] * Affine3d(coordTransform.inverse());

        geometry_msgs::Pose poseMsg = tf2::toMsg(poseR);


        cout << "Coord transform: " << coordTransformT.matrix() << endl;
        cout << "Loader pose: " << poseR.matrix() << endl;


        pcl::PointCloud<pcl::PointXYZ>::Ptr cloudR(new pcl::PointCloud<pcl::PointXYZ>());

        pcl::PointCloud<pcl::PointXYZ>::Ptr groundR(new pcl::PointCloud<pcl::PointXYZ>());

        pcl::PointCloud<pcl::PointXYZ>::Ptr notGroundR(new pcl::PointCloud<pcl::PointXYZ>());

        pcl::transformPointCloud(*cloud, *cloudR, coordTransformT);

        pcl::transformPointCloud(*ground, *groundR, coordTransformT);

        pcl::transformPointCloud(*notGround, *notGroundR, coordTransformT);


        sensor_msgs::PointCloud2 fullPointcloudMsg;
        pcl::toROSMsg(*cloudR, fullPointcloudMsg);

        sensor_msgs::PointCloud2 groundMsg;
        pcl::toROSMsg(*groundR, groundMsg);

        sensor_msgs::PointCloud2 obstaclesMsg;
        pcl::toROSMsg(*notGroundR, obstaclesMsg);

        isRequested = !WAIT_FOR_REQUESTS;
        timerTicked = false;

        throttleTimer = nodeHandle->createWallTimer(ros::WallDuration(THROTTLE_DURATION), throttleTimerCallback, true); // restart timer

        ros::Time time((position + 1) * 0.1);

        time = ros::Time::now();


        geometry_msgs::PoseStamped poseStampedMsg;
        poseStampedMsg.pose = poseMsg;
        poseStampedMsg.header.frame_id = "leftcam_optical";
        poseStampedMsg.header.stamp = time;

        posePublisher.publish(poseStampedMsg);

        fullPointcloudMsg.header.frame_id = "velodyne_optical";
        fullPointcloudMsg.header.stamp = time;
        obstaclesMsg.header.frame_id = "velodyne_optical";
        obstaclesMsg.header.stamp = time;
        groundMsg.header.frame_id = "velodyne_optical";
        groundMsg.header.stamp = time;

        ROS_INFO("Publishing point cloud of %lu points", cloudR->size());
        pointCloudPublisher.publish(fullPointcloudMsg);
        ROS_INFO("Publishing ground cloud of %lu points", groundR->size());
        obstaclesPublisher.publish(obstaclesMsg);
        ROS_INFO("Publishing obstacle cloud of %lu points", notGroundR->size());
        groundPublisher.publish(groundMsg);

        rosgraph_msgs::Clock clock;
        clock.clock = time;

        clockPublisher.publish(clock);

        std::stringstream msg;

        msg << "Loader: Published frame " << position << " at time" << time << std::endl;

        ROS_INFO("%s", msg.str().c_str());

        return true;

    }else{
        return false;
    }
}

void requestReceived(){
    ROS_INFO("Loader: scan request received");
    if(isRequested){
        return; // ignore too many requests
    }
    isRequested = true;
    if(timerTicked){
        publishNext();
    }
}

std::vector<std::string> scanFiles;
int currentPosition = -1;


void publishNext(){

    size_t validSearchStart = currentPosition;

    bool published = false;
    while(!published){

        //advance position
        currentPosition = (currentPosition + 1) % scanFiles.size();

        if(currentPosition == validSearchStart){
            ROS_FATAL("No valid scan in scan sequence");
            exit(EXIT_FAILURE);
        }
        published = publishScan(scanFiles, currentPosition);
    }
}

void requestMessageHandler(const std_msgs::Empty& msg){
    requestReceived();
}

/**
 * This tutorial demonstrates simple sending of messages over the ROS system.
 */
int main(int argc, char **argv)
{


    ros::init(argc, argv, "loader");


    ros::NodeHandle n;
    nodeHandle = &n;


    //load params

    ros::param::param<std::string>("~velodyne_dir", KITTI_SEQUENCE_DIR, "");

    ros::param::param<std::string>("~annot_dir", KITTI_SEQUENCE_ANNOT_DIR, "");

    ros::param::param<std::string>("~poses_file", POSES_FILE, "");

    ros::param::param<std::string>("~cache_dir", CACHE_DIR, "");

    ros::param::param<bool>("~enable_cache", ENABLE_CACHE, false);

    ros::param::param<double>("~throttle_duration", THROTTLE_DURATION, 0.0);

    ros::param::param<bool>("~wait_for_requests", WAIT_FOR_REQUESTS, true);

    ros::param::param<bool>("~initial_publish", INITIAL_PUBLISH, false);


    ros::param::param<int>("~start_frame", currentPosition, 0);


    cout << KITTI_SEQUENCE_DIR << std::endl;

    currentPosition--;


    tf::TransformListener listener;
    transformListener = &listener;

    scanFiles = listFiles(KITTI_SEQUENCE_DIR);
    std::vector<Eigen::Affine3d> loadedPoses = loadPoses(POSES_FILE);
    if(loadedPoses.size() == 0){
        // could not load poses, use unit poses instead

        for(size_t i=0; i< scanFiles.size(); i++){

            poses.push_back(Affine3d::Identity());
        }

    }else{
        for(size_t i=0; i < scanFiles.size(); i++){
            boost::filesystem::path path(scanFiles[i]);
            std::string pos = path.stem().string();
            long int intpos = strtol(pos.c_str(), (char**)0, 10);
            poses.push_back(loadedPoses[intpos]);
        }
    }

    if(scanFiles.size() == 0){
        ROS_FATAL("No input files found");
        return EXIT_FAILURE;
    }

    // load parameters

    ros::Subscriber requestSubscriber;

    pointCloudPublisher = n.advertise<sensor_msgs::PointCloud2>("point_cloud", 1);
    obstaclesPublisher = n.advertise<sensor_msgs::PointCloud2>("point_cloud_obstacles", 1);
    groundPublisher = n.advertise<sensor_msgs::PointCloud2>("point_cloud_ground", 1);

    clockPublisher = n.advertise<rosgraph_msgs::Clock>("clock", 1);

    posePublisher = n.advertise<geometry_msgs::PoseStamped>("pose", 1);

    if(WAIT_FOR_REQUESTS){
        if(INITIAL_PUBLISH){
            publishNext(); // publish immediately and start the throttle timer
        }else{
            throttleTimer = nodeHandle->createWallTimer(ros::WallDuration(THROTTLE_DURATION), throttleTimerCallback, true); // start throttle timer
            //register request handler
        }
        requestSubscriber = n.subscribe("request_scan", 1, requestMessageHandler);
    }else{
        if(INITIAL_PUBLISH){
            publishNext(); // publish immediately and start the loop
        }else{ // publish after first timer tick
            isRequested = true;
            throttleTimer = nodeHandle->createWallTimer(ros::WallDuration(THROTTLE_DURATION), throttleTimerCallback, true); // start loop
        }
    }


    ros::spin();

  return 0;
}
