//
// Created by da on 03/08/23.
//

#include "System.h"
#include "Frame.h"
#include "Track.h"
#include "LocalMapping.h"
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl-1.10/pcl/point_cloud.h>
#include <pcl-1.10/pcl/point_types.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <thread>
#include <fstream>

System::System(const string &strSettingFile)
{
    cv::FileStorage fsSettings(strSettingFile.c_str(), cv::FileStorage::READ);
    if (!fsSettings.isOpened()) {
        ROS_ERROR_STREAM("Failed to open settings file at: " << strSettingFile << endl);
        exit(-1);
    }

    mSonarTopic = (string) fsSettings["SonarTopic"];
    mOdomTopic = (string) fsSettings["OdomTopic"];
    mRange = (double) fsSettings["Range"];
    mGradientThreshold = (double) fsSettings["GradientThreshold"];
    mPyramidLayer = (int) fsSettings["PyramidLayer"];
    mFOV = ((double) fsSettings["FOV"] / 180) * M_PI;
    int loss_threshold = (int) fsSettings["LossThreshold"];
    double gradient_inlier_threshold = (double) fsSettings["GradientInlierThreshold"];
    cv::FileNode node = fsSettings["Tbs"];
    cv::Mat Tbs;
    mT_b_s = Eigen::Isometry3d::Identity();
    if (!node.empty()) {
        Tbs = node.mat();
        if (Tbs.rows != 4 || Tbs.cols != 4) {
            ROS_ERROR_STREAM("Tbs matrix have to be a 4x4 transformation matrix");
            exit(-1);
        }
    }
    else {
        ROS_ERROR_STREAM("Tbs matrix doesn't exist");
        exit(-1);
    }
    cv::cv2eigen(Tbs, mT_b_s.matrix());

    ROS_INFO_STREAM("SonarTopic: " << mSonarTopic << endl);
    ROS_INFO_STREAM("OdomTopic: " << mOdomTopic << endl);
    ROS_INFO_STREAM("Range: " << mRange << endl);
    ROS_INFO_STREAM("GradientThreshold: " << mGradientThreshold << endl);
    ROS_INFO_STREAM("PyramidLayer: " << mPyramidLayer << endl);
    ROS_INFO_STREAM("FOV: " << 180 * mFOV / M_PI << endl);
    ROS_INFO_STREAM("LossThreshold: " << loss_threshold << endl);
    ROS_INFO_STREAM("GradientInlierThreshold: " << gradient_inlier_threshold << endl);
    ROS_INFO_STREAM("Tbs: \n" << fixed << setprecision(9) << Tbs << endl);


    mpTracker = make_shared<Track>(mRange, mFOV, mPyramidLayer, loss_threshold,
                                   gradient_inlier_threshold, mT_b_s);
    shared_ptr<TrackState> track_state = make_shared<TrackUpToDate>(mpTracker);
    mpTracker->SetState(track_state);

    mpLocalMaper = make_shared<LocalMapping>(mpTracker);
    mpTracker->SetLocalMaper(mpLocalMaper);

    //start new thread
    mptLocalMaper = make_shared<thread>(&LocalMapping::Run, mpLocalMaper);

    mT_bw_b0 = Eigen::Isometry3d::Identity();
    // mOdom_Path.reserve(10000);

    ros::NodeHandle nh;
    mSonarPosePub = nh.advertise<geometry_msgs::PoseStamped>("/direct_sonar/pose_draw", 10);
    mOdomPub = nh.advertise<nav_msgs::Odometry>("/direct_sonar/odom_test", 1000);


    mPointCloudPub = nh.advertise<sensor_msgs::PointCloud2>("/direct_sonar/point_cloud", 1);


}

void System::frameLoad(const sensor_msgs::ImageConstPtr &image_msg,
                       const geometry_msgs::PoseStamped::ConstPtr &odom_msg)
{
    Eigen::Vector3d t(odom_msg->pose.position.y, odom_msg->pose.position.x, odom_msg->pose.position.z);
    Eigen::Quaterniond q(odom_msg->pose.orientation.w, odom_msg->pose.orientation.x,
                         odom_msg->pose.orientation.y, odom_msg->pose.orientation.z);
    double time = image_msg->header.stamp.toSec();
    Eigen::Isometry3d I = Eigen::Isometry3d::Identity();
    if (mT_bw_b0.matrix() == I.matrix()) {
        mT_bw_b0.setIdentity();
        mT_bw_b0.rotate(q);
        mT_bw_b0.pretranslate(t);
        mpLocalMaper->mT_bw_b0 = mT_bw_b0;
        // return;
    }
    Eigen::Isometry3d T_bw_bi = Eigen::Isometry3d::Identity();
    Eigen::AngleAxisd a(q);
    T_bw_bi.rotate(q);
    T_bw_bi.pretranslate(t);
    Eigen::Isometry3d T_b0_bi = mT_bw_b0.inverse() * T_bw_bi;
    mOdom_Path.insert(make_pair(time, T_b0_bi));
    Save();

    cv::Mat img = cv_bridge::toCvShare(image_msg, "bgr8")->image;
    cv::Mat down;
    cv::pyrDown(img, down, cv::Size(img.cols / 2, img.rows / 2));
    // Frame f(down, image_msg->header.stamp.toSec(), mRange, mFOV, mPyramidLayer, mGradientThreshold);
    Frame f(down, image_msg->header.stamp.toSec(), mRange, mFOV, T_b0_bi, mPyramidLayer,
            mGradientThreshold);
    ExtracPointCloud(down, time, f.mTheta, f.mTx, f.mTy, f.mScale);
    nav_msgs::Odometry odom_test;
    q = Eigen::Quaterniond(T_b0_bi.rotation());
    t = T_b0_bi.translation();
    odom_test.header.stamp = image_msg->header.stamp;
    odom_test.header.frame_id = "odom";
    odom_test.child_frame_id = "base_link";
    odom_test.pose.pose.position.x = t.x();
    odom_test.pose.pose.position.y = t.y();
    odom_test.pose.pose.position.z = t.z();
    odom_test.pose.pose.orientation.x = q.x();
    odom_test.pose.pose.orientation.y = q.y();
    odom_test.pose.pose.orientation.z = q.z();
    odom_test.pose.pose.orientation.w = q.w();
    mOdomPub.publish(odom_test);
    BroadcastTF(T_b0_bi,image_msg->header.stamp,"odom","base_link");
    // mTracker->TrackFrame2Frame(f);
    Eigen::Isometry3d T_s0_si = mpTracker->TrackFrame(f);

    Eigen::Isometry3d T_bw_bi_sonar = mT_bw_b0 * mT_b_s * T_s0_si * mT_b_s.inverse();

    geometry_msgs::PoseStamped pose;
    pose.header.stamp = ros::Time::now();
    pose.header.frame_id = "map";
    pose.pose.position.x = T_bw_bi_sonar.translation().x();
    pose.pose.position.y = T_bw_bi_sonar.translation().y();
    pose.pose.position.z = T_bw_bi_sonar.translation().z();
    q= Eigen::Quaterniond(T_bw_bi_sonar.rotation());
    pose.pose.orientation.x = q.x();
    pose.pose.orientation.y = q.y();
    pose.pose.orientation.z = q.z();
    pose.pose.orientation.w = q.w();
    mSonarPosePub.publish(pose);

}

void System::runRos()
{
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    image_transport::SubscriberFilter image_sub(it, mSonarTopic, 100,
                                                image_transport::TransportHints("compressed"));
    message_filters::Subscriber<geometry_msgs::PoseStamped> odom_sub(nh, mOdomTopic, 100);
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, geometry_msgs::PoseStamped> MySyncPolicy;
    message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), image_sub, odom_sub);
    sync.registerCallback(boost::bind(&System::frameLoad, this, _1, _2));



    // image_transport::SubscriberFilter image_sub2(it, mSonarTopic, 100);
    // message_filters::Subscriber<nav_msgs::Odometry> odom_sub2(nh, mOdomTopic, 100);
    // typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, nav_msgs::Odometry> MySyncPolicy2;
    // message_filters::Synchronizer<MySyncPolicy2> sync2(MySyncPolicy2(10), image_sub2, odom_sub2);
    // sync2.registerCallback(boost::bind(&System::frameLoad2, this, _1, _2));
    ros::spin();
}

void System::frameLoad2(const sensor_msgs::ImageConstPtr &image_msg,
                        const nav_msgs::Odometry_<allocator<void>>::ConstPtr &odom_msg)
{
    cv::Mat img = cv_bridge::toCvShare(image_msg, "bgr8")->image;
    Eigen::Vector3d t(odom_msg->pose.pose.position.x, odom_msg->pose.pose.position.y,
                      odom_msg->pose.pose.position.z);
    Eigen::Quaterniond q(odom_msg->pose.pose.orientation.w, odom_msg->pose.pose.orientation.x,
                         odom_msg->pose.pose.orientation.y, odom_msg->pose.pose.orientation.z);
    double time = image_msg->header.stamp.toSec();
    Eigen::Isometry3d I = Eigen::Isometry3d::Identity();
    if (mT_bw_b0.matrix() == I.matrix()) {
        mT_bw_b0.setIdentity();
        mT_bw_b0.rotate(q);
        mT_bw_b0.pretranslate(t);
        // return;
    }
    Eigen::Isometry3d T_bw_bi = Eigen::Isometry3d::Identity();
    Eigen::AngleAxisd a(q);
    T_bw_bi.rotate(q);
    T_bw_bi.pretranslate(t);
    Eigen::Isometry3d T_b0_bi = mT_bw_b0.inverse() * T_bw_bi;
    mOdom_Path.insert(make_pair(time, T_b0_bi));
    Save();
    Frame f(img, image_msg->header.stamp.toSec(), mRange, mFOV, mPyramidLayer, mGradientThreshold);
    // mTracker->TrackFrame2Frame(f);
    mpTracker->TrackFrame(f);
}

void System::Save()
{
    std::ofstream outfile;
    outfile.open(
            "/home/da/project/ros/direct_sonar_ws/src/direct_sonar_odometry/evaluation/stamped_groundtruth_gt.txt",
            std::ios_base::out);
    outfile << "#timestamp tx ty tz qx qy qz qw" << endl;
    for (auto time_pose: mOdom_Path) {
        Eigen::Isometry3d T_b0_bi = time_pose.second;
        Eigen::Isometry3d T_s0_si = mT_b_s.inverse() * T_b0_bi * mT_b_s;
        // Eigen::Vector3d t = T_s0_si.translation();
        // Eigen::Quaterniond q(T_s0_si.rotation());
        Eigen::Vector3d t = T_b0_bi.translation();
        Eigen::Quaterniond q(T_b0_bi.rotation());
        outfile << fixed << setprecision(12) << time_pose.first << " " << t.x() << " " << t.y() << " " << t.z()
                << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
    }
    outfile.close();

}

void System::ExtracPointCloud(const cv::Mat &img, double timestamp, double theta, double tx, double ty,
                              double scale)
{
    cv::Mat gray = img.clone();
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    // Set the intensity threshold
    int threshold = 200; // Change this as required

    // To store points whose intensity is greater than threshold
    std::vector<cv::Point> points;

    // Find points that are greater than threshold
    cv::Mat mask = gray > threshold; // This creates a binary mask
    cv::findNonZero(mask, points);

    // pcl::PointCloud<pcl::PointXYZ> cloud;
    // for (auto p: points) {
    //     Eigen::Vector2d p_pixel(p.x, p.y);
    //     Eigen::Vector3d p_3d = sonar2Dto3D(p_pixel, theta, tx, ty, scale);
    //     pcl::PointXYZ point;
    //     point.x = p_3d(0);
    //     point.y = p_3d(1);
    //     point.z = p_3d(2);
    //
    //     cloud.points.push_back(point);
    //
    // }
    // ROS_INFO_STREAM("PointCloud Size: "<<cloud.points.size());
    // sensor_msgs::PointCloud2 pc_msg;
    // pcl::toROSMsg(cloud, pc_msg);
    // pc_msg.header.frame_id = "base_link";
    // ros::Time time;
    // time.fromSec(timestamp);
    // pc_msg.header.stamp = time;
    // mPointCloudPub.publish(pc_msg);


}

void System::BroadcastTF(const Eigen::Isometry3d &T_c0_cj_orb,
                              const ros::Time &stamp,
                              const string &id,
                              const string &child_id)
{
    Eigen::Quaterniond rotation_q(T_c0_cj_orb.rotation());
    m_tb.sendTransform(
            tf::StampedTransform(
                    tf::Transform(tf::Quaternion(rotation_q.x(), rotation_q.y(), rotation_q.z(), rotation_q.w()),
                                  tf::Vector3(T_c0_cj_orb.translation().x(),
                                              T_c0_cj_orb.translation().y(),
                                              T_c0_cj_orb.translation().z())),
                    stamp, id, child_id));
}
