//
// Created by da on 09/08/23.
//

#include "LocalMapping.h"
#include "nanoflann.hpp"
#include "Track.h"
#include "Frame.h"
#include "MapPoint.h"
#include "OptimizationType.h"
#include <thread>
#include <chrono>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <sensor_msgs/Image.h>

#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl-1.10/pcl/point_cloud.h>
#include <pcl-1.10/pcl/point_types.h>
#include <pcl-1.10/pcl/common/transforms.h>

using namespace std;
using namespace nanoflann;

LocalMapping::LocalMapping(shared_ptr<Track> pTracker) : mpTracker(pTracker)
{
    ros::NodeHandle nh;
    mMarkerPub = nh.advertise<visualization_msgs::MarkerArray>("/direct_sonar/visualization_marker", 10);
    mPointCloudPub = nh.advertise<sensor_msgs::PointCloud2>("/direct_sonar/point_cloud", 10);
    mSaveService = nh.advertiseService("/direct_sonar/save_map", &LocalMapping::SaveMapCallback, this);
    mT_bw_b0.setIdentity();
};


LocalMapping::LocalMapping(shared_ptr<Track> pTracker, Eigen::Isometry3d T_bw_b0):mT_bw_b0(T_bw_b0)
{
    ros::NodeHandle nh;
    mPointCloudPub = nh.advertise<sensor_msgs::PointCloud2>("/direct_sonar/point_cloud", 10);
    mMarkerPub = nh.advertise<visualization_msgs::MarkerArray>("/direct_sonar/visualization_marker", 10);
    mSaveService = nh.advertiseService("/direct_sonar/save_map", &LocalMapping::SaveMapCallback, this);
};



void LocalMapping::Run()
{
    while (1) {
        if (!mProcessingQueue.empty()) {
            ProcessKeyFrame();
            if (mActiveFrameWindow.size() > 1) {
                OptimizeWindow();
                Visualize();
                NotifyTracker();
            }
        }
        else {
            this_thread::sleep_for(chrono::milliseconds(10));
        }
    }
}

void LocalMapping::NotifyTracker()
{
    shared_ptr<TrackState> state = make_shared<TrackToUpdate>(mpTracker);
    mpTracker->SetState(state);
}

void LocalMapping::InsertKeyFrame(shared_ptr<Frame> pF)
{
    unique_lock<shared_mutex> lock(mProcessingQueueMutex);
    mProcessingQueue.push_back(pF);
}

void LocalMapping::ProcessKeyFrame()
{
    unique_lock<shared_mutex> lock(mProcessingQueueMutex);
    if (mProcessingQueue.empty()) {
        return;
    }
    unique_lock<shared_mutex> lock2(mWindowMutex);
    auto pF_new = mProcessingQueue.front();
    // build more data association
    set<pair<double, double>> inliers_last, inliers_cur;
    map<pair<double, double>, pair<double, double>> association;
    for (auto rit = mActiveFrameWindow.rbegin(); rit != mActiveFrameWindow.rend(); ++rit) {
        shared_ptr<Frame> pF = *rit;
        // PredictCurrentPose(pF, mpCurrentFrame);
        PoseEstimationWindow2Frame(pF, pF_new, inliers_last, inliers_cur, association, mpTracker->mRange,
                                   mpTracker->mFOV);
        // PoseEstimationWindow2FramePyramid(pF, mpCurrentFrame, inliers_last, inliers_cur, association);
        pF->mInliers = inliers_last;
        pF_new->mInliers = inliers_cur;
        BuildAssociation(pF, pF_new, association, mpTracker->mGradientInlierThreshold);
    }


    mActiveFrameWindow.push_back(pF_new);
    if (mActiveFrameWindow.size() > mWindowSize) {
        mActiveFrameWindow.pop_front();
    }
    IncreaseMapPoints(pF_new);
    mProcessingQueue.pop_front();
}

void LocalMapping::OptimizeWindow()
{
    // unique_lock<shared_mutex> lock2(mpTracker->mTrackMutex);
    int max_frame_id = Frame::mFrameNum;
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType* linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3* solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);
    // optimizer.setVerbose(true);

    //add frame vertex
    map<shared_ptr<Frame>, VertexSonarPose*> all_frame_vertex;
    map<shared_ptr<MapPoint>, VertexSonarPoint*> all_landmark_vertex;
    vector<EdgeSE3Odom*> odom_edges;
    vector<EdgeSE3Sonar*> sonar_edges;
    {
        unique_lock<shared_mutex> lock(mWindowMutex);
        for (auto rit = mActiveFrameWindow.rbegin(); rit != mActiveFrameWindow.rend(); ++rit) {
            // for (auto pF: mActiveFrameWindow) {
            auto pF = *rit;
            Eigen::Isometry3d T_w_si = pF->GetPose();
            Eigen::Isometry3d T_si_w = T_w_si.inverse();
            g2o::SE3Quat est(T_si_w.rotation(), T_si_w.translation());
            VertexSonarPose* vSE3 = new VertexSonarPose();
            vSE3->setId(pF->mID);
            vSE3->setEstimate(T_w_si);
            optimizer.addVertex(vSE3);
            if (rit == mActiveFrameWindow.rend() - 1) {
                vSE3->setFixed(true);
                ROS_INFO_STREAM("Fix ID:" << pF->mID);
            }
            all_frame_vertex.insert(make_pair(pF, vSE3));

            for (auto obs: pF->GetObservationsF2L()) {
                //add landmark vertex
                shared_ptr<MapPoint> pMP = obs.second;
                if (all_landmark_vertex.count(pMP) == 0) {
                    VertexSonarPoint* vPoint = new VertexSonarPoint();
                    vPoint->setId(max_frame_id + 1 + pMP->mID);
                    vPoint->setEstimate(pMP->getPosition());
                    vPoint->setMarginalized(true);
                    if (rit == mActiveFrameWindow.rend() - 1) {
                        vPoint->setFixed(true);
                        // ROS_INFO_STREAM("Fix ID:" << pF->mID);
                    }
                    optimizer.addVertex(vPoint);
                    all_landmark_vertex.insert(make_pair(pMP, vPoint));
                }

                //add frame2landmark edge
                Eigen::Vector2d p_pixel_meas(obs.first.first, obs.first.second);//pixel measurement
                EdgeSE3Sonar* e = new EdgeSE3Sonar(p_pixel_meas, pF->mTheta, pF->mTx, pF->mTy,
                                                   pF->mScale);
                e->setVertex(0, vSE3);
                e->setVertex(1, all_landmark_vertex[pMP]);
                e->setInformation(Eigen::Matrix2d::Identity());
                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                rk->setDelta(5.0);
                // e->setRobustKernel(rk);
                optimizer.addEdge(e);
                sonar_edges.push_back(e);
            }
        }
        Eigen::Isometry3d T_b_s = mpTracker->mT_b_s;
        for (auto rit = mActiveFrameWindow.rbegin(); rit != mActiveFrameWindow.rend() - 1; ++rit) {
            auto pF = *rit;
            auto pF_pre = *(rit + 1);
            if (pF && pF_pre) {
                Eigen::Isometry3d T_b0_bj = pF->GetOdomPose();
                // Eigen::Isometry3d T_bj_b0 = T_b0_bj.inverse();
                Eigen::Isometry3d T_b0_bi = pF_pre->GetOdomPose();
                Eigen::Isometry3d T_bi_b0 = T_b0_bi.inverse();
                Eigen::Isometry3d T_si_sj = T_b_s.inverse() * T_bi_b0 * T_b0_bj * T_b_s;
                EdgeSE3Odom* e_odom = new EdgeSE3Odom(T_si_sj);
                e_odom->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(
                        pF_pre->mID)));
                e_odom->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(
                        pF->mID)));
                Eigen::Matrix<double, 6, 6> info = Eigen::Matrix<double, 6, 6>::Identity();
                info(0,0) = 1;
                info(1,1) = 1;
                info(2,2) = 200000;
                info(3,3) = 100;
                info(4,4) = 100;
                info(5,5) = 1;
                e_odom->setInformation(info);
                e_odom->computeError();
                optimizer.addEdge(e_odom);
                // ROS_INFO_STREAM("add odom edge: " << pF_pre->mID << " -> " << pF->mID << " error: "
                //                                   << e_odom->error().transpose()
                //                                   << "relative odom pose:\n" << T_si_sj.matrix());
                odom_edges.push_back(e_odom);

            }

            // Eigen::Isometry3d T_b_s = mpTracker->mT_b_s;
            // Eigen::Isometry3d T_bi_bj = T_b_s * T_si_sj * T_b_s.inverse();
        }
    }

    double sonar_chi2 =0, odom_chi2 = 0;
    for(auto e:sonar_edges){
        e->computeError();
        sonar_chi2+=e->chi2();
    }
    for(auto e:odom_edges){
        e->computeError();
        odom_chi2+=e->chi2();
    }
    ROS_INFO_STREAM("sonar chi2: " << sonar_chi2 << " odom chi2: " << odom_chi2);


    optimizer.initializeOptimization(0);
    optimizer.optimize(50);

    //recover optimized data
    {
        unique_lock<shared_mutex> lock(mWindowMutex);
        for (auto pF_pVF: all_frame_vertex) {
            auto pF = pF_pVF.first;
            Eigen::Isometry3d T_s0_si = pF_pVF.second->estimate();
            // Eigen::Isometry3d T_s0_si = T_si_w.inverse();
            pF->SetPose(T_s0_si);
            // ROS_INFO_STREAM("Frame: " << pF->mID << " pose:\n" << T_s0_si.matrix());

        }
        for (auto pMP_pVP: all_landmark_vertex) {
            auto pMP = pMP_pVP.first;
            Eigen::Vector3d pos = pMP_pVP.second->estimate();
            // pos(2) = 0;
            pMP->setPosition(pos);
        }
    }

}

shared_ptr<Frame> LocalMapping::GetLastFrameInWindow()
{
    shared_lock<shared_mutex> lock(mWindowMutex);
    return mActiveFrameWindow.back();
}

deque<shared_ptr<Frame>> LocalMapping::GetWindow()
{
    shared_lock<shared_mutex> lock(mWindowMutex);
    return mActiveFrameWindow;
}

void LocalMapping::Visualize()
{
    visualization_msgs::MarkerArray all_markers;


    //clear marker
    // visualization_msgs::Marker frame_marker;
    // frame_marker.header.frame_id = "map";
    // frame_marker.header.stamp = ros::Time::now();
    // frame_marker.ns = "frame";
    // frame_marker.id = 0;
    // frame_marker.type = visualization_msgs::Marker::CUBE;
    // frame_marker.action = visualization_msgs::Marker::DELETEALL;
    // frame_marker.pose.position.x = 0;
    // frame_marker.pose.position.y = 0;
    // frame_marker.pose.position.z = 0;
    // frame_marker.pose.orientation.x = 0;
    // frame_marker.pose.orientation.y = 0;
    // frame_marker.pose.orientation.z = 0;
    // frame_marker.pose.orientation.w = 1;
    // frame_marker.scale.x = 0.2;
    // frame_marker.scale.y = 2;
    // frame_marker.scale.z = 1;
    // frame_marker.color.a = 1.0;
    // frame_marker.color.r = 0.0;
    // frame_marker.color.g = 1.0;
    // frame_marker.color.b = 0.0;
    // all_markers.markers.push_back(frame_marker);



    //visualize landmark and data association
    visualization_msgs::Marker line_marker;
    line_marker.header.frame_id = "map";
    line_marker.header.stamp = ros::Time::now();
    line_marker.ns = "line";
    line_marker.id = 0;
    line_marker.type = visualization_msgs::Marker::LINE_LIST;
    line_marker.action = visualization_msgs::Marker::ADD;
    line_marker.scale.x = 0.01;
    line_marker.scale.y = 0.01;
    line_marker.scale.z = 0.01;
    line_marker.color.r = 0.0;
    line_marker.color.g = 0.0;
    line_marker.color.b = 1.0;
    line_marker.color.a = 0.1;


    std::set<int> point_id;

    pcl::PointCloud<pcl::PointXYZ> cloud;

    // visualize frame in the window
    Eigen::Isometry3d T_b_s = mpTracker->mT_b_s;
    auto pF_first = *(mActiveFrameWindow.rend()-1);
    for (auto pF: mActiveFrameWindow) {
        Eigen::Isometry3d T_s0_si = pF->GetPose();
        Eigen::Isometry3d T_b0_bi = T_b_s * T_s0_si * T_b_s.inverse();
        Eigen::Quaterniond q_b0_bi(T_b0_bi.rotation());
        visualization_msgs::Marker frame_marker;
        frame_marker.header.frame_id = "map";
        frame_marker.header.stamp = ros::Time::now();
        frame_marker.ns = "frame";
        frame_marker.id = pF->mID;
        frame_marker.type = visualization_msgs::Marker::CUBE;
        frame_marker.action = visualization_msgs::Marker::ADD;
        frame_marker.pose.position.x = T_b0_bi.translation().x();
        frame_marker.pose.position.y = T_b0_bi.translation().y();
        frame_marker.pose.position.z = T_b0_bi.translation().z();
        frame_marker.pose.orientation.x = q_b0_bi.x();
        frame_marker.pose.orientation.y = q_b0_bi.y();
        frame_marker.pose.orientation.z = q_b0_bi.z();
        frame_marker.pose.orientation.w = q_b0_bi.w();
        frame_marker.scale.x = 0.4;
        frame_marker.scale.y = 0.4;
        frame_marker.scale.z = 0.4;
        frame_marker.color.a = 0.7;
        if(pF->mID==pF_first->mID){
            frame_marker.color.r = 1.0;
            frame_marker.color.g = 0.1;
            frame_marker.color.b = 0.1;
        }
        else{
            frame_marker.color.r = 0.0;
            frame_marker.color.g = 0.0;
            frame_marker.color.b = 1.0;
        }
        // frame_marker.color.r = 0.0;
        // frame_marker.color.g = 1.0;
        // frame_marker.color.b = 0.0;
        all_markers.markers.push_back(frame_marker);


        auto obs = pF->GetObservationsF2L();
        for (auto ob: obs) {
            shared_ptr<MapPoint> pMP = ob.second;
            int obs_num = pMP->getObservationNum();
            Eigen::Vector3d pos = pMP->getPosition();
            visualization_msgs::Marker landmark_marker;
            landmark_marker.header.frame_id = "map";
            landmark_marker.header.stamp = ros::Time::now();
            landmark_marker.ns = "landmark";
            landmark_marker.id = pMP->mID;
            landmark_marker.type = visualization_msgs::Marker::SPHERE;
            landmark_marker.action = visualization_msgs::Marker::ADD;
            landmark_marker.pose.position.x = pos.x();
            landmark_marker.pose.position.y = pos.y();
            landmark_marker.pose.position.z = pos.z();
            landmark_marker.scale.x = 0.3;
            landmark_marker.scale.y = 0.3;
            landmark_marker.scale.z = 0.3;
            landmark_marker.color.a = 1.0;
            if (obs_num > 5) {
                landmark_marker.color.r = 0.0;
                landmark_marker.color.g = 1.0;
                landmark_marker.color.b = 0.0;

                all_markers.markers.push_back(landmark_marker);

                line_marker.points.push_back(frame_marker.pose.position);
                line_marker.points.push_back(landmark_marker.pose.position);
                if(point_id.count(pMP->mID)==0){
                    pcl::PointXYZ point;
                    point.x = pos(0);
                    point.y = pos(1);
                    point.z = pos(2);
                    cloud.points.push_back(point);
                    point_id.insert(pMP->mID);
                }
                if(mActiveMapPoints.count(pMP->mID)==0){
                    mActiveMapPoints.insert(make_pair(pMP->mID,pMP));
                }
            }
            // else {
            //     landmark_marker.color.r = 1.0;
            //     landmark_marker.color.g = 0.0;
            //     landmark_marker.color.b = 0.0;
            // }


        }
    }
    all_markers.markers.push_back(line_marker);

    mMarkerPub.publish(all_markers);

    ROS_INFO_STREAM("PointCloud Size: "<<cloud.points.size());
    // convert pointcloud from sonar frame to odom frame
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZ>);
    Eigen::Isometry3d T_bw_s0 = mT_bw_b0 * mpTracker->mT_b_s;
    pcl::transformPointCloud(cloud,*transformed_cloud,T_bw_s0.matrix().cast<float>());


    sensor_msgs::PointCloud2 pc_msg;
    pcl::toROSMsg(*transformed_cloud, pc_msg);
    pc_msg.header.frame_id = "odom";

    pc_msg.header.stamp = line_marker.header.stamp;
    mPointCloudPub.publish(pc_msg);


}

void LocalMapping::Reset()
{
    unique_lock<shared_mutex> lock(mWindowMutex);
    unique_lock<shared_mutex> lock2(mProcessingQueueMutex);
    mActiveFrameWindow.clear();
    mProcessingQueue.clear();
}

void LocalMapping::PoseEstimationWindow2Frame(shared_ptr<Frame> pF_pre, shared_ptr<Frame> pF,
                                              set<pair<double, double>> &inliers_last,
                                              set<pair<double, double>> &inliers_current,
                                              map<pair<double, double>, pair<double, double>> &association,
                                              double range, double fov)
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

    double scale = (cur_gray.rows) / range;
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

        EdgeSE3SonarDirect* edge = new EdgeSE3SonarDirect(p3d, theta, tx, ty, scale, range, fov,
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
        if (e->error().norm() > mpTracker->mGradientInlierThreshold * 0.5 || (e->error().norm() == 0)) {
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
    if(association.size()>mpTracker->mLossThreshold){
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

bool LocalMapping::SaveMapCallback(std_srvs::EmptyRequest &req, std_srvs::EmptyResponse &res)
{
    pcl::PointCloud<pcl::PointXYZ> cloud;
    for(auto it: mActiveMapPoints){
        auto pos = it.second->getPosition();
        pcl::PointXYZ p(pos.x(),pos.y(),pos.z());
        cloud.points.push_back(p);
    }
    ROS_INFO_STREAM("PointCloud Size: "<<cloud.points.size());
    // convert pointcloud from sonar frame to odom frame
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZ>);
    Eigen::Isometry3d T_bw_s0 = mT_bw_b0 * mpTracker->mT_b_s;
    pcl::transformPointCloud(cloud,*transformed_cloud,T_bw_s0.matrix().cast<float>());


    sensor_msgs::PointCloud2 pc_msg;
    pcl::toROSMsg(*transformed_cloud, pc_msg);
    pc_msg.header.frame_id = "map";

    pc_msg.header.stamp = ros::Time::now();
    mPointCloudPub.publish(pc_msg);
    return true;
}

void LocalMapping::PubMap()
{
    pcl::PointCloud<pcl::PointXYZ> cloud;
    for(auto it: mActiveMapPoints){
        auto pos = it.second->getPosition();
        pcl::PointXYZ p(pos.x(),pos.y(),pos.z());
        cloud.points.push_back(p);
    }
    ROS_INFO_STREAM("PointCloud Size: "<<cloud.points.size());
    // convert pointcloud from sonar frame to odom frame
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZ>);
    Eigen::Isometry3d T_bw_s0 = mT_bw_b0 * mpTracker->mT_b_s;
    pcl::transformPointCloud(cloud,*transformed_cloud,T_bw_s0.matrix().cast<float>());


    sensor_msgs::PointCloud2 pc_msg;
    pcl::toROSMsg(*transformed_cloud, pc_msg);
    pc_msg.header.frame_id = "map";

    pc_msg.header.stamp = ros::Time::now();
    mPointCloudPub.publish(pc_msg);
    // return true;
}
