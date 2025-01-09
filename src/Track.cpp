//
// Created by da on 03/08/23.
//

#include "Track.h"
#include "Frame.h"
#include "OptimizationType.h"
#include "MapPoint.h"
#include "LocalMapping.h"
#include <cv_bridge.h>
#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <sensor_msgs/Image.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Path.h>
#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace nanoflann;
using namespace std;

Track::Track(double range, double fov, int pyramid_layer, int loss_threshold, double gradient_threshold,
             Eigen::Isometry3d &T_b_s) : mRange(range), mFOV(fov), mPyramidLayer(pyramid_layer),
                                         mT_b_s(T_b_s), mLossThreshold(loss_threshold),
                                         mGradientInlierThreshold(gradient_threshold)
{
    ros::NodeHandle nh;
    mImagePub = nh.advertise<sensor_msgs::Image>("/direct_sonar/image", 10);
    mPosePub = nh.advertise<geometry_msgs::PoseStamped>("/direct_sonar/pose", 10);
    mPathPub = nh.advertise<nav_msgs::Path>("/direct_sonar/path", 10);
    mOdomPub = nh.advertise<geometry_msgs::PoseStamped>("/direct_sonar/odom", 10);
    mOdomPathPub = nh.advertise<nav_msgs::Path>("/direct_sonar/odom_path", 10);
}

void Track::TrackFrame2Frame(const Frame &f)
{
    mpCurrentFrame = make_shared<Frame>(f);
    if (!mpLastFrame) {
        ROS_INFO_STREAM("mpLastFrame is empty");
        mPath.insert(make_pair(mpCurrentFrame->mTimestamp, mT_w_sj));
        mOdomPath.insert(make_pair(mpCurrentFrame->mTimestamp, mpCurrentFrame->GetOdomPose()));
        mpLastFrame = mpCurrentFrame;
        return;
    }
    Eigen::Isometry3d Tsj_si = Eigen::Isometry3d::Identity();
    for (int layer = mPyramidLayer - 1; layer >= 0; layer--) {
        cv::Mat img = mpCurrentFrame->mPyramid[layer];
        cv::Mat pre_img = mpLastFrame->mPyramid[layer];
        set<pair<double, double>> inliers_last, inliers_cur;
        map<pair<double, double>, pair<double, double>> association;
        PoseEstimationFrame2Frame(pre_img, img, Tsj_si, mpLastFrame->mKeyPoints, layer, inliers_last,
                                  inliers_cur, association);
    }
    mT_w_sj = mT_w_sj * Tsj_si.inverse();


    mPath.insert(make_pair(mpCurrentFrame->mTimestamp, mT_w_sj));
    mOdomPath.insert(make_pair(mpCurrentFrame->mTimestamp, mpCurrentFrame->GetOdomPose()));
    mpLastFrame = mpCurrentFrame;
    SavePath("");
}

Eigen::Isometry3d Track::TrackFrame(const Frame &f)
{
    unique_lock<shared_mutex> lock(mStateMutex);
    mState->TrackFrame(f);
    mState = make_shared<TrackUpToDate>(shared_from_this());
    ROS_INFO_STREAM("Frame[" << mpCurrentFrame->mID << "]" << " inlier: "
                             << mpCurrentFrame->GetObservationsF2L().size());
    return mpCurrentFrame->GetPose();
}

void Track::TrackFromLastFrame(const Frame &f)
{
    ROS_INFO_STREAM("TrackFromLastFrame");
    unique_lock<shared_mutex> lock(mTrackMutex);
    mpCurrentFrame = make_shared<Frame>(f);
    //first frame
    if (!mpLastFrame) {
        ROS_INFO_STREAM("mpLastFrame is empty");
        mpCurrentFrame->SetPose(Eigen::Isometry3d::Identity());
        mpCurrentFrame->mInliers = mpCurrentFrame->mKeyPoints;
        InitializeMap();
        mPath.insert(make_pair(mpCurrentFrame->mTimestamp, mpCurrentFrame->GetPose()));
        mOdomPath.insert(make_pair(mpCurrentFrame->mTimestamp, mpCurrentFrame->GetOdomPose()));
        mpLocalMaper->InsertKeyFrame(mpCurrentFrame);
        mpLastFrame = mpCurrentFrame;
        return;
    }
    // loss long time from last keyframe
    shared_ptr<Frame> kf_pre = mpLocalMaper->GetLastFrameInWindow();
    if (mpCurrentFrame->mTimestamp - kf_pre->mTimestamp >= 6.0) {
        Reset(kf_pre);
        return;
    }


    // set initial pose by frame2frame track
    set<pair<double, double>> inliers_last, inliers_cur;
    map<pair<double, double>, pair<double, double>> association;
    // shared_ptr<Frame> pF_last = mActiveFrameWindow.back();
    PredictCurrentPose(mpLastFrame, mpCurrentFrame);
    Eigen::Isometry3d T_s0_scur = mpCurrentFrame->GetPose();
    Eigen::Isometry3d T_s0_spre = mpLastFrame->GetPose();
    Eigen::Isometry3d Tsicur_sipre = T_s0_scur.inverse() * T_s0_spre;
    // Eigen::Isometry3d Tsicur_sipre = Eigen::Isometry3d::Identity();
    for (int layer = mPyramidLayer - 1; layer >= 0; layer--) {
        cv::Mat img = mpCurrentFrame->mPyramid[layer];
        cv::Mat pre_img = mpLastFrame->mPyramid[layer];
        PoseEstimationFrame2Frame(pre_img, img, Tsicur_sipre, mpLastFrame->mKeyPoints, layer,
                                  inliers_last, inliers_cur, association);
    }
    mT_w_sj = mpLastFrame->GetPose() * Tsicur_sipre.inverse();
    mpCurrentFrame->SetPose(mT_w_sj);
    // mpLastFrame->mInliers = inliers_last;
    // mpCurrentFrame->mInliers = inliers_cur;
    // BuildAssociation(mpLastFrame, mpCurrentFrame, association);

    // track local window
    inliers_last.clear();
    inliers_cur.clear();
    association.clear();
    auto ActiveFrameWindow = mpLocalMaper->GetWindow();
    int window_index = 0;
    for (auto rit = ActiveFrameWindow.rbegin(); rit != ActiveFrameWindow.rend(); ++rit) {
        shared_ptr<Frame> pF = *rit;
        // PredictCurrentPose(pF, mpCurrentFrame);
        PoseEstimationWindow2Frame(pF, mpCurrentFrame, inliers_last, inliers_cur, association);
        // PoseEstimationWindow2FramePyramid(pF, mpCurrentFrame, inliers_last, inliers_cur, association);
        pF->mInliers = inliers_last;
        mpCurrentFrame->mInliers = inliers_cur;
        BuildAssociation(pF, mpCurrentFrame, association, mLossThreshold);
        // ROS_INFO_STREAM(
        //         "tracked points: " << mpCurrentFrame->GetObservationsF2L().size() << " from Frame: "
        //                            << pF->mID);
        if (mpCurrentFrame->GetObservationsF2L().size() >= mLossThreshold || window_index > 2) {
            break;
        }
        window_index++;
    }

    // check whether track sucssess, if not use odom pose
    if (mpCurrentFrame->GetObservationsF2L().size() < mLossThreshold) {
        PredictCurrentPose(mpLastFrame, mpCurrentFrame);
    }


    if (NeedNewKeyFrame(mpCurrentFrame)) {
        InsertKeyFrame(mpCurrentFrame);
    }
    mpLastFrame = mpCurrentFrame;
    // mPath.insert(make_pair(mpCurrentFrame->mTimestamp, mpCurrentFrame->GetPose()));
    // SavePath("");
    DrawTrack();
    PublishPose();

}

void Track::TrackFromWindow(const Frame &f)
{
    ROS_INFO_STREAM("TrackFromWindow");
    unique_lock<shared_mutex> lock(mTrackMutex);
    mpCurrentFrame = make_shared<Frame>(f);
    mpLastFrame = mpLocalMaper->GetLastFrameInWindow();
    if (!mpLastFrame) {
        ROS_INFO_STREAM("mpLastFrame is empty");
        mpCurrentFrame->SetPose(Eigen::Isometry3d::Identity());
        mpCurrentFrame->mInliers = mpCurrentFrame->mKeyPoints;
        InitializeMap();
        mPath.insert(make_pair(mpCurrentFrame->mTimestamp, mpCurrentFrame->GetPose()));
        mOdomPath.insert(make_pair(mpCurrentFrame->mTimestamp, mpCurrentFrame->GetOdomPose()));
        mpLocalMaper->InsertKeyFrame(mpCurrentFrame);
        mpLastFrame = mpCurrentFrame;
        return;
    }

    // set initial pose by frame2frame track
    set<pair<double, double>> inliers_last, inliers_cur;
    map<pair<double, double>, pair<double, double>> association;
    // shared_ptr<Frame> pF_last = mActiveFrameWindow.back();
    PredictCurrentPose(mpLastFrame, mpCurrentFrame);
    Eigen::Isometry3d T_s0_scur = mpCurrentFrame->GetPose();
    Eigen::Isometry3d T_s0_spre = mpLastFrame->GetPose();
    Eigen::Isometry3d Tsicur_sipre = T_s0_scur.inverse() * T_s0_spre;
    // Eigen::Isometry3d Tsicur_sipre = Eigen::Isometry3d::Identity();
    for (int layer = mPyramidLayer - 1; layer >= 0; layer--) {
        cv::Mat img = mpCurrentFrame->mPyramid[layer];
        cv::Mat pre_img = mpLastFrame->mPyramid[layer];
        PoseEstimationFrame2Frame(pre_img, img, Tsicur_sipre, mpLastFrame->mKeyPoints, layer,
                                  inliers_last, inliers_cur, association);
    }
    mT_w_sj = mpLastFrame->GetPose() * Tsicur_sipre.inverse();
    mpCurrentFrame->SetPose(mT_w_sj);
    // mpLastFrame->mInliers = inliers_last;
    // mpCurrentFrame->mInliers = inliers_cur;
    // BuildAssociation(mpLastFrame, mpCurrentFrame, association);

    // track local window
    inliers_last.clear();
    inliers_cur.clear();
    association.clear();
    auto ActiveFrameWindow = mpLocalMaper->GetWindow();
    int window_index = 0;
    for (auto rit = ActiveFrameWindow.rbegin(); rit != ActiveFrameWindow.rend(); ++rit) {
        shared_ptr<Frame> pF = *rit;
        // PredictCurrentPose(pF, mpCurrentFrame);
        PoseEstimationWindow2Frame(pF, mpCurrentFrame, inliers_last, inliers_cur, association);
        // PoseEstimationWindow2FramePyramid(pF, mpCurrentFrame, inliers_last, inliers_cur, association);
        pF->mInliers = inliers_last;
        mpCurrentFrame->mInliers = inliers_cur;
        BuildAssociation(pF, mpCurrentFrame, association, mLossThreshold);
        // ROS_INFO_STREAM(
        //         "tracked points: " << mpCurrentFrame->GetObservationsF2L().size() << " from Frame: "
        //                            << pF->mID);
        if (mpCurrentFrame->GetObservationsF2L().size() >= mLossThreshold || window_index > 2) {
            break;
        }
        window_index++;
    }

    if (mpCurrentFrame->GetObservationsF2L().size() < mLossThreshold) {
        PredictCurrentPose(mpLastFrame, mpCurrentFrame);
    }


    if (NeedNewKeyFrame(mpCurrentFrame)) {
        InsertKeyFrame(mpCurrentFrame);

    }
    mPath.insert(make_pair(mpCurrentFrame->mTimestamp, mpCurrentFrame->GetPose()));
    mOdomPath.insert(make_pair(mpCurrentFrame->mTimestamp, mpCurrentFrame->GetOdomPose()));
    SavePath("");
    mpLastFrame = mpCurrentFrame;
    DrawTrack();
    PublishPose();

}

bool Track::PoseEstimationFrame2Frame(cv::Mat &pre_img, cv::Mat &img, Eigen::Isometry3d &Tsj_si,
                                      set<pair<double, double>> &obs, int layer,
                                      set<pair<double, double>> &inliers_last,
                                      set<pair<double, double>> &inliers_current,
                                      map<pair<double, double>, pair<double, double>> &association)
{
    //convert rgb to gray
    cv::Mat pre_gray;
    cv::cvtColor(pre_img, pre_gray, cv::COLOR_BGR2GRAY);
    cv::Mat cur_gray;
    cv::cvtColor(img, cur_gray, cv::COLOR_BGR2GRAY);

    double scale = (img.rows) / mRange;
    double theta = -0.5 * M_PI;
    double tx = 0.5 * img.cols;
    double ty = img.rows;

    // setup g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 1>> DirectBlock;
    DirectBlock::LinearSolverType* linearSolver = new g2o::LinearSolverDense<DirectBlock::PoseMatrixType>();
    DirectBlock* solver_ptr = new DirectBlock(linearSolver);
    // g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton( solver_ptr ); // G-N
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
            solver_ptr); // L-M
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
    pose->setEstimate(g2o::SE3Quat(Tsj_si.rotation(), Tsj_si.translation()));
    pose->setId(0);
    optimizer.addVertex(pose);
    vector<EdgeSE3SonarDirect*> all_edges;
    int edge_id = 0;

    map<int, pair<double, double>> id_pixel;
    for (auto it: obs) {
        Eigen::Vector2d p_pixel(it.first, it.second);
        p_pixel = p_pixel * pow(0.5, layer);
        Eigen::Vector3d p3d = sonar2Dto3D(p_pixel, theta, tx, ty, scale);
        // double grayscale = getPixelValue(pre_gray, p_pixel.x(), p_pixel.y());
        double grayscale = getPatternPixelValue(pre_gray, p_pixel.x(), p_pixel.y());

        id_pixel.insert(make_pair(edge_id, it));

        EdgeSE3SonarDirect* edge = new EdgeSE3SonarDirect(p3d, theta, tx, ty, scale, mRange, mFOV,
                                                          cur_gray);
        edge->setVertex(0, pose);
        edge->setMeasurement(grayscale);
        edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
        edge->setId(edge_id++);
        //set robust kernel
        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
        rk->setDelta(10.0);
        // edge->setRobustKernel(rk);
        optimizer.addEdge(edge);
        all_edges.push_back(edge);
    }
    // cout << "edges in graph: " << optimizer.edges().size() << endl;
    // ROS_INFO_STREAM("edges in graph: " << optimizer.edges().size());

    optimizer.initializeOptimization();
    optimizer.optimize(100);
    //save edge chi2 to file
    ofstream f_chi2;
    f_chi2.open(
            "/home/da/project/ros/direct_sonar_ws/src/direct_sonar_odometry/debug/opt_frame2frame.txt",
            ios::out);
    for (auto e: all_edges) {
        e->computeError();
        f_chi2 << "edge[" << e->id() << "]: " << e->error().norm() << endl;
        if (e->error().norm() > mGradientInlierThreshold || (e->error().norm() == 0)) {
            //remove from id_pixel
            id_pixel.erase(e->id());
        }
    }
    // ROS_INFO_STREAM("inlier number: " << id_pixel.size());
    f_chi2.close();

    inliers_last.clear();
    for (auto it: id_pixel) {
        inliers_last.insert(it.second);
    }


    Eigen::Isometry3d T = pose->estimate();
    Tsj_si = T;

    inliers_current.clear();
    association.clear();
    for (auto it: inliers_last) {
        Eigen::Vector2d p_pixel(it.first, it.second);
        p_pixel = p_pixel * pow(0.5, layer);
        Eigen::Vector3d p = sonar2Dto3D(p_pixel, theta, tx, ty, scale);
        Eigen::Vector3d p2 = Tsj_si * p;
        Eigen::Vector2d pixel_now = sonar3Dto2D(p2, theta, tx, ty, scale);
        inliers_current.insert(make_pair(pixel_now.x(), pixel_now.y()));
        association.insert(make_pair(it, make_pair(pixel_now.x(), pixel_now.y())));
    }

    // cout << "Tcw=\n" << Tsj_si.matrix() << endl;
    //convert to angle axis
    // Eigen::AngleAxisd a(Tsj_siq.rotation());
    // cout << "rotation axis: " << a.axis().transpose() << " angle: " << a.angle() << endl;

    // plot the feature points
    // cv::Mat img_show(img.rows * 2, img.cols * 2, CV_8UC3);
    // pre_img.copyTo(img_show(cv::Rect(0, 0, img.cols, img.rows)));
    // img.copyTo(img_show(cv::Rect(0, img.rows, img.cols, img.rows)));
    //
    // pre_img.copyTo(img_show(cv::Rect(img.cols, 0, img.cols, img.rows)));
    // img.copyTo(img_show(cv::Rect(img.cols, img.rows, img.cols, img.rows)));
    // if (layer == 0) {
    //     for (auto it: inliers_last) {
    //         Eigen::Vector2d p_pixel(it.first, it.second);
    //         p_pixel = p_pixel * pow(0.5, layer);
    //         Eigen::Vector3d p = sonar2Dto3D(p_pixel, theta, tx, ty, scale);
    //         // Eigen::Vector3d p = m.pos_world;
    //         Eigen::Vector2d pixel_prev = sonar3Dto2D(p, theta, tx, ty, scale);
    //         Eigen::Vector3d p2 = Tsj_si * p;
    //         Eigen::Vector2d pixel_now = sonar3Dto2D(p2, theta, tx, ty, scale);
    //         if (pixel_now(0, 0) < 0 || pixel_now(0, 0) >= img.cols || pixel_now(1, 0) < 0 ||
    //             pixel_now(1, 0) >= img.rows) {
    //             continue;
    //         }
    //
    //         float b = 0;
    //         float g = 250;
    //         float r = 0;
    //         img_show.ptr<uchar>(pixel_prev(1, 0))[int(pixel_prev(0, 0)) * 3] = b;
    //         img_show.ptr<uchar>(pixel_prev(1, 0))[int(pixel_prev(0, 0)) * 3 + 1] = g;
    //         img_show.ptr<uchar>(pixel_prev(1, 0))[int(pixel_prev(0, 0)) * 3 + 2] = r;
    //
    //         img_show.ptr<uchar>(pixel_now(1, 0) + img.rows)[int(pixel_now(0, 0)) * 3] = b;
    //         img_show.ptr<uchar>(pixel_now(1, 0) + img.rows)[int(pixel_now(0, 0)) * 3 + 1] = g;
    //         img_show.ptr<uchar>(pixel_now(1, 0) + img.rows)[int(pixel_now(0, 0)) * 3 + 2] = r;
    //         cv::circle(img_show, cv::Point2d(pixel_prev(0, 0), pixel_prev(1, 0)), 1, cv::Scalar(b, g, r),
    //                    1);
    //         cv::circle(img_show, cv::Point2d(pixel_now(0, 0), pixel_now(1, 0) + img.rows), 1,
    //                    cv::Scalar(b, g, r), 1);
    //
    //     }
    //     //get ros time
    //     auto t = ros::Time::now();
    //     stringstream ss;
    //     ss << "/home/da/project/ros/direct_sonar_ws/src/direct_sonar_odometry/results/"
    //        << mpLastFrame->mID << "_" << mpCurrentFrame->mID << ".png";
    //     // cv::imshow("result", img_show);
    //     // cv::waitKey(0);
    //     cv::imwrite(ss.str(), img_show);
    //
    // }

    return inliers_current.size() > 100;
}


void Track::PoseEstimationWindow2Frame(shared_ptr<Frame> pF_pre, shared_ptr<Frame> pF,
                                       set<pair<double, double>> &inliers_last,
                                       set<pair<double, double>> &inliers_current,
                                       map<pair<double, double>, pair<double, double>> &association)
{
    //convert rgb to gray
    cv::Mat pre_gray;
    cv::cvtColor(pF_pre->mImg, pre_gray, cv::COLOR_BGR2GRAY);
    cv::Mat cur_gray;
    cv::cvtColor(pF->mImg, cur_gray, cv::COLOR_BGR2GRAY);
    //init pose
    Eigen::Isometry3d T_w_si = pF_pre->GetPose();
    Eigen::Isometry3d T_w_sj = pF->GetPose();
    Eigen::Isometry3d T_sj_si = T_w_sj.inverse() * T_w_si;

    double scale = (cur_gray.rows) / mRange;
    double theta = -0.5 * M_PI;
    double tx = 0.5 * cur_gray.cols;
    double ty = cur_gray.rows;

    // setup g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 1>> DirectBlock;
    DirectBlock::LinearSolverType* linearSolver = new g2o::LinearSolverDense<DirectBlock::PoseMatrixType>();
    DirectBlock* solver_ptr = new DirectBlock(linearSolver);
    // g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton( solver_ptr ); // G-N
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
            solver_ptr); // L-M
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
    pose->setEstimate(g2o::SE3Quat(T_sj_si.rotation(), T_sj_si.translation()));
    pose->setId(0);
    optimizer.addVertex(pose);
    vector<EdgeSE3SonarDirect*> all_edges;
    int edge_id = 0;

    map<int, pair<double, double>> id_pixel;
    auto obs = pF_pre->GetObservationsF2L();
    for (auto it: obs) {
        Eigen::Vector2d p_pixel(it.first.first, it.first.second);
        Eigen::Vector3d p3d = sonar2Dto3D(p_pixel, theta, tx, ty, scale);
        // double grayscale = getPixelValue(pre_gray, p_pixel.x(), p_pixel.y());
        double grayscale = getPatternPixelValue(pre_gray, p_pixel.x(), p_pixel.y());

        id_pixel.insert(make_pair(edge_id, it.first));

        EdgeSE3SonarDirect* edge = new EdgeSE3SonarDirect(p3d, theta, tx, ty, scale, mRange, mFOV,
                                                          cur_gray);
        edge->setVertex(0, pose);
        edge->setMeasurement(grayscale);
        edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
        edge->setId(edge_id++);
        //set robust kernel
        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
        rk->setDelta(10.0);
        // edge->setRobustKernel(rk);
        optimizer.addEdge(edge);
        all_edges.push_back(edge);
    }
    // cout << "edges in graph: " << optimizer.edges().size() << endl;
    // ROS_INFO_STREAM("edges in graph: " << optimizer.edges().size());

    optimizer.initializeOptimization();
    optimizer.optimize(100);


    Eigen::Isometry3d T = pose->estimate();
    T_sj_si = T;



    // cout << "Tcw=\n" << T sj_si.matrix() << endl;
    //convert to angle axis
    // Eigen::AngleAxisd a(T sj_si.rotation());
    // cout << "rotation axis: " << a.axis().transpose() << " angle: " << a.angle() << endl;


    //save edge chi2 to file
    ofstream f_chi2;
    f_chi2.open("/home/da/project/ros/direct_sonar_ws/src/direct_sonar_odometry/debug/opt.txt",
                ios::out);

    for (auto e: all_edges) {
        // e->computeError();
        f_chi2 << "edge[" << e->id() << "]: " << e->error().norm() << endl;
        if (e->error().norm() > mGradientInlierThreshold || (e->error().norm() == 0)) {
            //remove from id_pixel
            id_pixel.erase(e->id());
        }
    }
    // ROS_INFO_STREAM("inlier number: " << id_pixel.size());
    f_chi2.close();

    inliers_last.clear();
    for (auto it: id_pixel) {
        inliers_last.insert(it.second);
    }

    inliers_current.clear();
    association.clear();
    for (auto it: inliers_last) {
        Eigen::Vector2d p_pixel(it.first, it.second);
        Eigen::Vector3d p = sonar2Dto3D(p_pixel, theta, tx, ty, scale);
        Eigen::Vector3d p2 = T_sj_si * p;
        Eigen::Vector2d pixel_now = sonar3Dto2D(p2, theta, tx, ty, scale);
        inliers_current.insert(make_pair(pixel_now.x(), pixel_now.y()));
        association.insert(make_pair(it, make_pair(pixel_now.x(), pixel_now.y())));
    }

    //set pose
    if(pF_pre==mpLocalMaper->GetWindow().back()){
        T_w_si = pF_pre->GetPose();
        T_w_sj = T_w_si * T_sj_si.inverse();
        pF->SetPose(T_w_sj);
    }

    // plot the feature points
    // cv::Mat img_show(pF->mImg.rows * 2, pF->mImg.cols, CV_8UC3);
    // pF_pre->mImg.copyTo(img_show(cv::Rect(0, 0, pF->mImg.cols, pF->mImg.rows)));
    // pF->mImg.copyTo(img_show(cv::Rect(0, pF->mImg.rows, pF->mImg.cols, pF->mImg.rows)));
    //
    // // pF_pre->mImg.copyTo(img_show(cv::Rect(pF->mImg.cols, 0, pF->mImg.cols, pF->mImg.rows)));
    // // pF->mImg.copyTo(img_show(cv::Rect(pF->mImg.cols, pF->mImg.rows, pF->mImg.cols, pF->mImg.rows)));
    //
    // for (auto it: inliers_last) {
    //     Eigen::Vector2d p_pixel(it.first, it.second);
    //     Eigen::Vector3d p = sonar2Dto3D(p_pixel, theta, tx, ty, scale);
    //     // Eigen::Vector3d p = m.pos_world;
    //     Eigen::Vector2d pixel_prev = sonar3Dto2D(p, theta, tx, ty, scale);
    //     Eigen::Vector3d p2 = T_sj_si * p;
    //     Eigen::Vector2d pixel_now = sonar3Dto2D(p2, theta, tx, ty, scale);
    //     if (pixel_now(0, 0) < 0 || pixel_now(0, 0) >= pF->mImg.cols || pixel_now(1, 0) < 0 ||
    //         pixel_now(1, 0) >= pF->mImg.rows) {
    //         continue;
    //     }
    //
    //     float b = 0;
    //     float g = 250;
    //     float r = 0;
    //     img_show.ptr<uchar>(pixel_prev(1, 0))[int(pixel_prev(0, 0)) * 3] = b;
    //     img_show.ptr<uchar>(pixel_prev(1, 0))[int(pixel_prev(0, 0)) * 3 + 1] = g;
    //     img_show.ptr<uchar>(pixel_prev(1, 0))[int(pixel_prev(0, 0)) * 3 + 2] = r;
    //
    //     img_show.ptr<uchar>(pixel_now(1, 0) + pF->mImg.rows)[int(pixel_now(0, 0)) * 3] = b;
    //     img_show.ptr<uchar>(pixel_now(1, 0) + pF->mImg.rows)[int(pixel_now(0, 0)) * 3 + 1] = g;
    //     img_show.ptr<uchar>(pixel_now(1, 0) + pF->mImg.rows)[int(pixel_now(0, 0)) * 3 + 2] = r;
    //     cv::circle(img_show, cv::Point2d(pixel_prev(0, 0), pixel_prev(1, 0)), 2, cv::Scalar(b, g, r), 1);
    //     cv::circle(img_show, cv::Point2d(pixel_now(0, 0), pixel_now(1, 0) + pF->mImg.rows), 2,
    //                cv::Scalar(b, g, r), 1);
    //
    // }
    // //get ros time
    // auto t = ros::Time::now();
    // stringstream ss;
    // ss << "/home/da/project/ros/direct_sonar_ws/src/direct_sonar_odometry/results/WINDOW_" << pF->mID
    //    << "_" << pF_pre->mID << ".png";
    // // cv::imshow("result", img_show);
    // // cv::waitKey(0);
    // cv::imwrite(ss.str(), img_show);


}

void Track::PoseEstimationWindow2FramePyramid(shared_ptr<Frame> pF_pre, shared_ptr<Frame> pF,
                                              set<pair<double, double>> &inliers_last,
                                              set<pair<double, double>> &inliers_current,
                                              map<pair<double, double>, pair<double, double>> &association)
{

    //init pose
    Eigen::Isometry3d T_w_si = pF_pre->GetPose();
    Eigen::Isometry3d T_w_sj = pF->GetPose();
    Eigen::Isometry3d T_sj_si = T_w_sj.inverse() * T_w_si;

    // Eigen::Isometry3d Tsj_si = Eigen::Isometry3d::Identity();
    for (int layer = mPyramidLayer - 1; layer >= 0; layer--) {
        cv::Mat img = pF->mPyramid[layer];
        cv::Mat pre_img = pF_pre->mPyramid[layer];
        set<pair<double, double>> obs;
        auto obsF2L = pF_pre->GetObservationsF2L();
        for (auto F_L: obsF2L) {
            obs.insert(F_L.first);
        }
        PoseEstimationFrame2Frame(pre_img, img, T_sj_si, obs, layer, inliers_last, inliers_current,
                                  association);
    }



    //set pose
    T_w_si = pF_pre->GetPose();
    T_w_sj = T_w_si * T_sj_si.inverse();
    pF->SetPose(T_w_sj);

    // cout << "Tcw=\n" << T sj_si.matrix() << endl;
    //convert to angle axis
    // Eigen::AngleAxisd a(T sj_si.rotation());
    // cout << "rotation axis: " << a.axis().transpose() << " angle: " << a.angle() << endl;

    // plot the feature points
    cv::Mat img_show(pF->mImg.rows * 2, pF->mImg.cols * 2, CV_8UC3);
    pF_pre->mImg.copyTo(img_show(cv::Rect(0, 0, pF->mImg.cols, pF->mImg.rows)));
    pF->mImg.copyTo(img_show(cv::Rect(0, pF->mImg.rows, pF->mImg.cols, pF->mImg.rows)));

    pF_pre->mImg.copyTo(img_show(cv::Rect(pF->mImg.cols, 0, pF->mImg.cols, pF->mImg.rows)));
    pF->mImg.copyTo(img_show(cv::Rect(pF->mImg.cols, pF->mImg.rows, pF->mImg.cols, pF->mImg.rows)));

    for (auto it: inliers_last) {
        Eigen::Vector2d p_pixel(it.first, it.second);
        Eigen::Vector3d p = sonar2Dto3D(p_pixel, pF->mTheta, pF->mTx, pF->mTy, pF->mScale);
        // Eigen::Vector3d p = m.pos_world;
        Eigen::Vector2d pixel_prev = sonar3Dto2D(p, pF->mTheta, pF->mTx, pF->mTy, pF->mScale);
        Eigen::Vector3d p2 = T_sj_si * p;
        Eigen::Vector2d pixel_now = sonar3Dto2D(p2, pF->mTheta, pF->mTx, pF->mTy, pF->mScale);
        if (pixel_now(0, 0) < 0 || pixel_now(0, 0) >= pF->mImg.cols || pixel_now(1, 0) < 0 ||
            pixel_now(1, 0) >= pF->mImg.rows) {
            continue;
        }

        float b = 0;
        float g = 250;
        float r = 0;
        img_show.ptr<uchar>(pixel_prev(1, 0))[int(pixel_prev(0, 0)) * 3] = b;
        img_show.ptr<uchar>(pixel_prev(1, 0))[int(pixel_prev(0, 0)) * 3 + 1] = g;
        img_show.ptr<uchar>(pixel_prev(1, 0))[int(pixel_prev(0, 0)) * 3 + 2] = r;

        img_show.ptr<uchar>(pixel_now(1, 0) + pF->mImg.rows)[int(pixel_now(0, 0)) * 3] = b;
        img_show.ptr<uchar>(pixel_now(1, 0) + pF->mImg.rows)[int(pixel_now(0, 0)) * 3 + 1] = g;
        img_show.ptr<uchar>(pixel_now(1, 0) + pF->mImg.rows)[int(pixel_now(0, 0)) * 3 + 2] = r;
        cv::circle(img_show, cv::Point2d(pixel_prev(0, 0), pixel_prev(1, 0)), 2, cv::Scalar(b, g, r), 1);
        cv::circle(img_show, cv::Point2d(pixel_now(0, 0), pixel_now(1, 0) + pF->mImg.rows), 2,
                   cv::Scalar(b, g, r), 1);

    }
    //get ros time
    auto t = ros::Time::now();
    stringstream ss;
    ss << "/home/da/project/ros/direct_sonar_ws/src/direct_sonar_odometry/results/WINDOW_" << pF->mID
       << "_" << pF_pre->mID << ".png";
    // cv::imshow("result", img_show);
    // cv::waitKey(0);
    cv::imwrite(ss.str(), img_show);


}

void Track::SavePath(string file_path)
{
    std::ofstream outfile;
    outfile.open(
            "/home/da/project/ros/direct_sonar_ws/src/direct_sonar_odometry/evaluation/stamped_traj_estimate.txt",
            std::ios_base::out);
    outfile << "#timestamp tx ty tz qx qy qz qw" << endl;
    for (auto s_p: mPath) {
        Eigen::Isometry3d T_s0_si = s_p.second;
        Eigen::Isometry3d T_b0_bi = mT_b_s * T_s0_si * mT_b_s.inverse();
        Eigen::Vector3d t = T_b0_bi.translation();
        Eigen::Quaterniond q(T_b0_bi.rotation());
        outfile << fixed << setprecision(12) << s_p.first << " " << t.x() << " " << t.y() << " " << t.z()
                << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
    }
    outfile.close();

    // stringstream ss;
    // ss << "/home/da/project/ros/direct_sonar_ws/src/direct_sonar_odometry/debug/"
    //    << mActiveFrameWindow.size() << "_" << mActiveFrameWindow.front()->mID << "_obs.txt";
    // outfile.open(ss.str(), std::ios_base::out);
    // auto obs_f2l = mActiveFrameWindow.front()->GetObservationsF2L();
    // for (auto f_l: obs_f2l) {
    //     outfile << f_l.first.first << " " << f_l.first.second << endl;
    // }
    // outfile.close();
}

void Track::InitializeMap()
{
    for (auto p: mpCurrentFrame->mKeyPoints) {
        Eigen::Vector2d p_pixel(p.first, p.second);
        Eigen::Vector3d p3d = sonar2Dto3D(p_pixel, mpCurrentFrame->mTheta, mpCurrentFrame->mTx,
                                          mpCurrentFrame->mTy, mpCurrentFrame->mScale);
        Eigen::Isometry3d T_w_sj = mpCurrentFrame->GetPose();
        Eigen::Vector3d p_w = T_w_sj * p3d;
        shared_ptr<MapPoint> p_mp = make_shared<MapPoint>(p_w);

        mMapPoints.insert(make_pair(p_mp->mID, p_mp));

        AssociateFrameToLandmark(mpCurrentFrame, p_mp, p);


    }
}

void
AssociateFrameToLandmark(shared_ptr<Frame> pF, shared_ptr<MapPoint> pMP, pair<double, double> p_pixel)
{
    pF->AddObservation(pMP, p_pixel);
    pMP->AddObservation(pF);

}

void
UnassociateFrameToLandmark(shared_ptr<Frame> pF, shared_ptr<MapPoint> pMP, pair<double, double> p_pixel)
{
    pF->RemoveObservation(pMP->mID);
    pMP->RemoveObservation(pF->mID);
}

void BuildAssociation(shared_ptr<Frame> pF_last, shared_ptr<Frame> pF_cur,
                      map<pair<double, double>, pair<double, double>> &association, int loss_threshold)
{
    if (pF_cur->GetObservationsF2L().size() + association.size() < loss_threshold || association.size()<20) {
        ROS_INFO_STREAM(
                "skip association, too few inlier: Frame" << pF_last->mID << "-> Frame" << pF_cur->mID);
        return;
    }
    for (auto it: pF_last->mInliers) {
        auto obs_F2L = pF_last->GetObservationsF2L();
        auto key_cur = association[it];
        shared_ptr<MapPoint> pm = obs_F2L[it];
        AssociateFrameToLandmark(pF_cur, pm, key_cur);
    }
}

void IncreaseMapPoints(shared_ptr<Frame> pF)
{
    // use inlier pixel position build a KD-tree
    auto obs = pF->GetObservationsF2L();
    PointCloud cloud;
    for (auto it: obs) {
        Point2D point;
        point.x = it.first.first;
        point.y = it.first.second;
        cloud.pts.push_back(point);
    }
    KDTree index(2 /*dim*/, cloud, KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
    index.buildIndex();


    int increase_num = 0;
    for (auto it: pF->mKeyPoints) {
        const size_t num_results = 1;
        size_t ret_index;
        double out_dist_sqr;
        nanoflann::KNNResultSet<double> resultSet(num_results);
        resultSet.init(&ret_index, &out_dist_sqr);

        Point2D point;
        point.x = it.first;
        point.y = it.second;
        double query_pt[2] = {point.x, point.y};
        index.findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParameters(10));
        if (sqrt(out_dist_sqr) > 5) {
            Eigen::Vector2d p_pixel(it.first, it.second);
            Eigen::Vector3d p3d = sonar2Dto3D(p_pixel, pF->mTheta, pF->mTx, pF->mTy, pF->mScale);
            Eigen::Isometry3d T_w_sj = pF->GetPose();
            Eigen::Vector3d p_w = T_w_sj * p3d;
            shared_ptr<MapPoint> p_mp = make_shared<MapPoint>(p_w);

            // mMapPoints.insert(make_pair(p_mp->mID, p_mp));

            AssociateFrameToLandmark(pF, p_mp, it);
            increase_num++;

        }
        // else {
        //     ROS_INFO_STREAM("reject new point distance: " << sqrt(out_dist_sqr));
        // }
    }
    ROS_INFO_STREAM("increase map points: " << increase_num << "from Frame:" << pF->mID);


}

bool Track::NeedNewKeyFrame(shared_ptr<Frame> pF)
{
    // if (pF->GetObservationsF2L().size() < 0.5 * mLossThreshold) {
    //     return false;
    // }
    shared_ptr<Frame> pLastKF = mpLocalMaper->GetLastFrameInWindow();
    if (pF->mTimestamp - pLastKF->mTimestamp >= 5.0) {
        return true;
    }
    else if (pF->GetObservationsF2L().size() < mpCurrentFrame->mKeyPoints.size() * 0.5 &&(pF->mTimestamp - pLastKF->mTimestamp >= 1.0)) {
        return true;
    }
    else {
        return false;
    }

}

void Track::InsertKeyFrame(shared_ptr<Frame> pF)
{
    mpLocalMaper->InsertKeyFrame(pF);
}

void Track::DrawTrack()
{
    auto obs = mpCurrentFrame->GetObservationsF2L();
    cv::Mat out = mpCurrentFrame->mImg.clone();
    for (auto it: obs) {
        float b = 0;
        float g = 250;
        float r = 0;
        cv::circle(out, cv::Point2d(it.first.first, it.first.second), 2, cv::Scalar(b, g, r), 1);
    }
    //show image
    // cv::imshow("track", out);
    stringstream ss;
    ss << "/home/da/project/ros/direct_sonar_ws/src/direct_sonar_odometry/results/Track_"
       << mpCurrentFrame->mID << ".png";
    // cv::imwrite(ss.str(), out);
    //publish image
    cv_bridge::CvImage cvImage;
    cvImage.header.stamp = ros::Time::now();
    cvImage.header.frame_id = "map";
    cvImage.encoding = "bgr8";
    cvImage.image = out;
    mImagePub.publish(cvImage.toImageMsg());
    // cv::waitKey(1);
}


void Track::SetState(const shared_ptr<TrackState> &State)
{
    unique_lock<shared_mutex> lock(mStateMutex);
    mState = State;
}

void Track::SetLocalMaper(const shared_ptr<LocalMapping> &pLocalMaper)
{
    mpLocalMaper = pLocalMaper;
}

void Track::PredictCurrentPose(shared_ptr<Frame> f_pre, shared_ptr<Frame> f_cur)
{
    Eigen::Isometry3d T_b0_bipre = f_pre->GetOdomPose();
    Eigen::Isometry3d T_b0_bicur = f_cur->GetOdomPose();
    Eigen::Isometry3d T_bipre_bicur = T_b0_bipre.inverse() * T_b0_bicur;
    Eigen::Isometry3d T_sipre_sicur = mT_b_s.inverse() * T_bipre_bicur * mT_b_s;
    Eigen::Isometry3d T_s0_sipre = f_pre->GetPose();
    Eigen::Isometry3d T_s0_si_prediction = T_s0_sipre * T_sipre_sicur;
    f_cur->SetPose(T_s0_si_prediction);
}

void Track::PublishPose()
{
    Eigen::Isometry3d T_s0_si = mpCurrentFrame->GetPose();
    Eigen::Isometry3d T_b0_bi = mT_b_s * T_s0_si * mT_b_s.inverse();
    geometry_msgs::PoseStamped pose;
    pose.header.stamp = ros::Time::now();
    pose.header.frame_id = "map";
    pose.pose.position.x = T_b0_bi.translation().x();
    pose.pose.position.y = T_b0_bi.translation().y();
    pose.pose.position.z = T_b0_bi.translation().z();
    Eigen::Quaterniond q(T_b0_bi.rotation());
    pose.pose.orientation.x = q.x();
    pose.pose.orientation.y = q.y();
    pose.pose.orientation.z = q.z();
    pose.pose.orientation.w = q.w();

    mPosePub.publish(pose);

    nav_msgs::Path path;
    path.header = pose.header;
    for (auto time_pose: mPath) {
        T_s0_si = time_pose.second;
        T_b0_bi = mT_b_s * T_s0_si * mT_b_s.inverse();
        pose.pose.position.x = T_b0_bi.translation().x();
        pose.pose.position.y = T_b0_bi.translation().y();
        pose.pose.position.z = T_b0_bi.translation().z();
        q = Eigen::Quaterniond(T_b0_bi.rotation());
        pose.pose.orientation.x = q.x();
        pose.pose.orientation.y = q.y();
        pose.pose.orientation.z = q.z();
        pose.pose.orientation.w = q.w();
        path.poses.push_back(pose);
    }
    mPathPub.publish(path);

    T_b0_bi = mpCurrentFrame->GetOdomPose();
    pose.pose.position.x = T_b0_bi.translation().x();
    pose.pose.position.y = T_b0_bi.translation().y();
    pose.pose.position.z = T_b0_bi.translation().z();
    q = Eigen::Quaterniond(T_b0_bi.rotation());
    pose.pose.orientation.x = q.x();
    pose.pose.orientation.y = q.y();
    pose.pose.orientation.z = q.z();
    pose.pose.orientation.w = q.w();
    mOdomPub.publish(pose);

    path.poses.clear();
    for (auto time_pose: mOdomPath) {
        T_b0_bi = time_pose.second;
        pose.pose.position.x = T_b0_bi.translation().x();
        pose.pose.position.y = T_b0_bi.translation().y();
        pose.pose.position.z = T_b0_bi.translation().z();
        q = Eigen::Quaterniond(T_b0_bi.rotation());
        pose.pose.orientation.x = q.x();
        pose.pose.orientation.y = q.y();
        pose.pose.orientation.z = q.z();
        pose.pose.orientation.w = q.w();
        path.poses.push_back(pose);
    }
    mOdomPathPub.publish(path);


}

void Track::Reset(shared_ptr<Frame> f_pre)
{
    PredictCurrentPose(f_pre, mpCurrentFrame);
    if (mpCurrentFrame->mKeyPoints.size() < mLossThreshold) {
        PublishPose();
        DrawTrack();
        return;
    }
    mpCurrentFrame->mInliers = mpCurrentFrame->mKeyPoints;
    InitializeMap();
    mPath.insert(make_pair(mpCurrentFrame->mTimestamp, mpCurrentFrame->GetPose()));
    mOdomPath.insert(make_pair(mpCurrentFrame->mTimestamp, mpCurrentFrame->GetOdomPose()));
    mpLocalMaper->Reset();
    mpLocalMaper->InsertKeyFrame(mpCurrentFrame);
    mpLastFrame = mpCurrentFrame;
    ROS_INFO_STREAM("Reset Done");
    return;
}


void TrackToUpdate::TrackFrame(const Frame &f)
{
    mpTracker->TrackFromWindow(f);
}

void TrackUpToDate::TrackFrame(const Frame &f)
{
    mpTracker->TrackFromLastFrame(f);
}
