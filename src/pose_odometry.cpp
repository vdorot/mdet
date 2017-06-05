#include "ros/ros.h"
#include "std_msgs/String.h"
#include "geometry_msgs/Pose.h"
#include <tf/transform_broadcaster.h>
#include <sstream>
#include <tf/transform_listener.h>

#include "tf2_geometry_msgs/tf2_geometry_msgs.h"
#include "tf2_eigen/tf2_eigen.h"


tf::TransformBroadcaster *tfBroadcaster;
tf::TransformListener *tfListener;

void poseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg)
{

    char *TARGET_FRAME = "/odom";

    char *WORLD_FRAME = "/map";

    try {
        ros::Time trTime = msg->header.stamp;
        while(!tfListener->waitForTransform(msg->header.frame_id, TARGET_FRAME, trTime, ros::Duration(20.0))){

        }

        tf::StampedTransform stTr;
        tfListener->lookupTransform(msg->header.frame_id, TARGET_FRAME, trTime, stTr);

        tf::Stamped<tf::Pose> pin;
        tf::Stamped<tf::Pose> pout;
        tf::poseStampedMsgToTF(*msg, pin);

        pout.setData(stTr.inverse() * (pin * stTr));

        pout.stamp_ = pin.stamp_;
        pout.frame_id_ = WORLD_FRAME;

        geometry_msgs::PoseStamped poseMsg;
        tf::poseStampedTFToMsg(pout, poseMsg);

        geometry_msgs::TransformStamped tfMsg;

        tfMsg.header = poseMsg.header;
        tfMsg.child_frame_id = TARGET_FRAME;

        tfMsg.transform.rotation = poseMsg.pose.orientation;
        tfMsg.transform.translation.x = poseMsg.pose.position.x;
        tfMsg.transform.translation.y = poseMsg.pose.position.y;
        tfMsg.transform.translation.z = poseMsg.pose.position.z;

        tfBroadcaster->sendTransform(tfMsg);

        std::cout << "Published odom transform for time" << trTime << std::endl;

    }catch(tf2::TransformException &ex){
        ROS_WARN("Could NOT transform turtle2 to turtle1: %s", ex.what());
    }

}

int main(int argc, char **argv)
{

    ros::init(argc, argv, "pose_odometry");
    ros::NodeHandle n;

    tf::TransformBroadcaster br;
    tfBroadcaster = &br;
    tf::TransformListener ls;
    tfListener = &ls;

    ros::Subscriber sub = n.subscribe("pose", 1, poseCallback);
    ros::spin();

    return 0;
}