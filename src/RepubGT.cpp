#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>

Eigen::Isometry3d T_bw_b0_GT;
bool isGTInit = false;
nav_msgs::Path mPath;
ros::Publisher mGTPub;
ros::Publisher mGTPosePub;
ros::Publisher mGTPathPub;

Eigen::Isometry3d T_bw_b0_odom;
bool isOdomInit = false;
nav_msgs::Path mPathOdom;
ros::Publisher mOdomPub;
ros::Publisher mOdomPosePub;
ros::Publisher mOdomPathPub;

void GTCallback(const nav_msgs::Odometry::ConstPtr &msg)
{
    Eigen::Vector3d t(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z);
    Eigen::Quaterniond q(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x,
                         msg->pose.pose.orientation.y, msg->pose.pose.orientation.z);
    Eigen::Isometry3d T_bw_bj;
    T_bw_bj.setIdentity();
    T_bw_bj.rotate(q);
    T_bw_bj.pretranslate(t);
    if (!isGTInit) {
        T_bw_b0_GT = T_bw_bj;
        isGTInit = true;
        return;
    }
    Eigen::Isometry3d T_b0_bj = T_bw_b0_GT.inverse() * T_bw_bj;
    t = T_b0_bj.translation();
    q = Eigen::Quaterniond(T_b0_bj.rotation());
    nav_msgs::Odometry gt_odom_msg;
    gt_odom_msg.header = msg->header;
    gt_odom_msg.header.frame_id = "map";
    gt_odom_msg.child_frame_id = "base_link";
    gt_odom_msg.pose.pose.position.x = t.x();
    gt_odom_msg.pose.pose.position.y = t.y();
    gt_odom_msg.pose.pose.position.z = t.z();
    gt_odom_msg.pose.pose.orientation.x = q.x();
    gt_odom_msg.pose.pose.orientation.y = q.y();
    gt_odom_msg.pose.pose.orientation.z = q.z();
    gt_odom_msg.pose.pose.orientation.w = q.w();
    mGTPub.publish(gt_odom_msg);

    geometry_msgs::PoseStamped pose;
    pose.header = gt_odom_msg.header;
    pose.pose = gt_odom_msg.pose.pose;
    mGTPosePub.publish(pose);

    mPath.poses.push_back(pose);
    mGTPathPub.publish(mPath);
}

void GTPoseCallback(const geometry_msgs::PoseStamped::ConstPtr &msg)
{
    Eigen::Vector3d t(msg->pose.position.y, msg->pose.position.x, msg->pose.position.z);
    Eigen::Quaterniond q(msg->pose.orientation.w, msg->pose.orientation.x,
                         msg->pose.orientation.y, msg->pose.orientation.z);
    Eigen::Isometry3d T_bw_bj;
    T_bw_bj.setIdentity();
    T_bw_bj.rotate(q);
    T_bw_bj.pretranslate(t);
    if (!isGTInit) {
        T_bw_b0_GT = T_bw_bj;
        isGTInit = true;
        return;
    }
    Eigen::Isometry3d T_b0_bj = T_bw_b0_GT.inverse() * T_bw_bj;
    t = T_b0_bj.translation();
    q = Eigen::Quaterniond(T_b0_bj.rotation());
    nav_msgs::Odometry gt_odom_msg;
    gt_odom_msg.header = msg->header;
    gt_odom_msg.header.frame_id = "map";
    gt_odom_msg.child_frame_id = "base_link";
    gt_odom_msg.pose.pose.position.x = t.x();
    gt_odom_msg.pose.pose.position.y = t.y();
    gt_odom_msg.pose.pose.position.z = t.z();
    gt_odom_msg.pose.pose.orientation.x = q.x();
    gt_odom_msg.pose.pose.orientation.y = q.y();
    gt_odom_msg.pose.pose.orientation.z = q.z();
    gt_odom_msg.pose.pose.orientation.w = q.w();
    mGTPub.publish(gt_odom_msg);

    geometry_msgs::PoseStamped pose;
    pose.header = gt_odom_msg.header;
    pose.pose = gt_odom_msg.pose.pose;
    mGTPosePub.publish(pose);

    mPath.poses.push_back(pose);
    mGTPathPub.publish(mPath);
}

void OdomCallback(const nav_msgs::Odometry::ConstPtr &msg)
{
    Eigen::Vector3d t(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z);
    Eigen::Quaterniond q(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x,
                         msg->pose.pose.orientation.y, msg->pose.pose.orientation.z);
    Eigen::Isometry3d T_bw_bj;
    T_bw_bj.setIdentity();
    T_bw_bj.rotate(q);
    T_bw_bj.pretranslate(t);
    if (!isOdomInit) {
        T_bw_b0_odom = T_bw_bj;
        isOdomInit = true;
        return;
    }
    Eigen::Isometry3d T_b0_bj = T_bw_b0_odom.inverse() * T_bw_bj;
    t = T_b0_bj.translation();
    q = Eigen::Quaterniond(T_b0_bj.rotation());
    nav_msgs::Odometry odom_msg;
    odom_msg.header = msg->header;
    odom_msg.header.frame_id = "map";
    odom_msg.child_frame_id = "base_link";
    odom_msg.pose.pose.position.x = t.x();
    odom_msg.pose.pose.position.y = t.y();
    odom_msg.pose.pose.position.z = t.z();
    odom_msg.pose.pose.orientation.x = q.x();
    odom_msg.pose.pose.orientation.y = q.y();
    odom_msg.pose.pose.orientation.z = q.z();
    odom_msg.pose.pose.orientation.w = q.w();
    mOdomPub.publish(odom_msg);

    geometry_msgs::PoseStamped pose;
    pose.header = odom_msg.header;
    pose.pose = odom_msg.pose.pose;
    mOdomPosePub.publish(pose);

    mPathOdom.poses.push_back(pose);
    mOdomPathPub.publish(mPathOdom);
}

void Odom2Callback(const geometry_msgs::PoseStamped::ConstPtr &msg)
{
    Eigen::Vector3d t(msg->pose.position.y, msg->pose.position.x, msg->pose.position.z);
    Eigen::Quaterniond q(msg->pose.orientation.w, msg->pose.orientation.x,
                         msg->pose.orientation.y, msg->pose.orientation.z);
    Eigen::Isometry3d T_bw_bj;
    T_bw_bj.setIdentity();
    T_bw_bj.rotate(q);
    T_bw_bj.pretranslate(t);
    if (!isOdomInit) {
        T_bw_b0_odom = T_bw_bj;
        isOdomInit = true;
        return;
    }
    Eigen::Isometry3d T_b0_bj = T_bw_b0_odom.inverse() * T_bw_bj;
    t = T_b0_bj.translation();
    q = Eigen::Quaterniond(T_b0_bj.rotation());
    nav_msgs::Odometry odom_msg;
    odom_msg.header = msg->header;
    odom_msg.header.frame_id = "map";
    odom_msg.child_frame_id = "base_link";
    odom_msg.pose.pose.position.x = t.x();
    odom_msg.pose.pose.position.y = t.y();
    odom_msg.pose.pose.position.z = t.z();
    odom_msg.pose.pose.orientation.x = q.x();
    odom_msg.pose.pose.orientation.y = q.y();
    odom_msg.pose.pose.orientation.z = q.z();
    odom_msg.pose.pose.orientation.w = q.w();
    mOdomPub.publish(odom_msg);

    geometry_msgs::PoseStamped pose;
    pose.header = odom_msg.header;
    pose.pose = odom_msg.pose.pose;
    mOdomPosePub.publish(pose);

    mPathOdom.poses.push_back(pose);
    mOdomPathPub.publish(mPathOdom);
}

int main(int argc, char** argv)
{
    // Initialize the ROS node
    ros::init(argc, argv, "odom_subscriber_node");
    ros::NodeHandle nh;

    // Create a subscriber for the odometry topic
    ros::Subscriber odom_sub = nh.subscribe("/rexrov/pose_gt", 10, GTCallback);
    ros::Subscriber gt_pose_sub = nh.subscribe("/pose_gt", 10, GTPoseCallback);

    mGTPub = nh.advertise<nav_msgs::Odometry>("/gt_repub/gt_odom", 10);
    mGTPathPub = nh.advertise<nav_msgs::Path>("/gt_repub/gt_path", 10);
    mGTPosePub = nh.advertise<geometry_msgs::PoseStamped>("/gt_repub/gt_pose", 10);

    // Create a subscriber for the odometry topic
    ros::Subscriber odom_sub_odom = nh.subscribe("/odometry/filtered", 10, OdomCallback);
    ros::Subscriber odom_sub_odom2 = nh.subscribe("/odom_pose", 10, Odom2Callback);

    mOdomPub = nh.advertise<nav_msgs::Odometry>("/gt_repub/odom", 10);
    mOdomPathPub = nh.advertise<nav_msgs::Path>("/gt_repub/odom_path", 10);
    mOdomPosePub = nh.advertise<geometry_msgs::PoseStamped>("/gt_repub/odom_pose", 10);

    mPath.header.stamp = ros::Time::now();
    mPath.header.frame_id = "map";
    mPathOdom.header = mPath.header;
    //init pose
    T_bw_b0_GT.setIdentity();
    T_bw_b0_odom.setIdentity();

    // Spin to allow the callback function to run when new messages arrive
    ros::spin();

    return 0;
}

