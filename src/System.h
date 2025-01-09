//
// Created by da on 03/08/23.
//

#ifndef SRC_SYSTEM_H
#define SRC_SYSTEM_H

#include<string>
#include <thread>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <image_transport/subscriber_filter.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Path.h>
#include <tf/transform_broadcaster.h>

using namespace std;

class Track;
class LocalMapping;
class System
{
public:
    System(const string &strSettingFile);

    void frameLoad(const sensor_msgs::ImageConstPtr &image_msg,
                   const geometry_msgs::PoseStamped::ConstPtr &odom_msg);

    void frameLoad2(const sensor_msgs::ImageConstPtr &image_msg,
                    const nav_msgs::Odometry::ConstPtr &odom_msg);

    void ExtracPointCloud(const cv::Mat &img, double timestamp, double theta, double tx, double ty,
                          double scale);
    void BroadcastTF(const Eigen::Isometry3d &T_c0_cj_orb,
                             const ros::Time &stamp,
                             const string &id,
                             const string &child_id);
    void Save();

    void runRos();

private:
    string mSonarTopic;
    string mOdomTopic;
    double mRange;
    double mGradientThreshold;
    int mPyramidLayer;
    double mFOV;

    //odometry initial pose
    Eigen::Isometry3d mT_bw_b0;
    //extrinsic parameter sonar to body
    Eigen::Isometry3d mT_b_s;
    map<double, Eigen::Isometry3d> mOdom_Path;
    shared_ptr<Track> mpTracker;
    shared_ptr<LocalMapping> mpLocalMaper;
    shared_ptr<thread> mptLocalMaper;
    ros::Publisher mSonarPosePub;

    ros::Publisher mOdomPub;
    ros::Publisher mPointCloudPub;
    tf::TransformBroadcaster m_tb;
    // ros::Publisher mOdomPub;
};


#endif //SRC_SYSTEM_H
