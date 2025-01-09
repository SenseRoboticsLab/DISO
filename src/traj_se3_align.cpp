//
// Created by da on 15/08/23.
//
#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <map>
#include <chrono>
#include <ctime>
#include <climits>
#include <string>
#include <sstream>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <string>

#include <sophus/geometry.hpp>

#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>

using namespace std;
using namespace g2o;

class EdgeSE3Align : public BaseMultiEdge<6, SE3Quat>
{
public:
    EdgeSE3Align()
    {
        resize(3);
    }

    void computeError() override
    {
        const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[0]);
        const VertexSE3Expmap* v2 = static_cast<const VertexSE3Expmap*>(_vertices[1]);
        const VertexSE3Expmap* v3 = static_cast<const VertexSE3Expmap*>(_vertices[2]);

        Eigen::Isometry3d T_s0_si = v1->estimate();
        Eigen::Isometry3d T_e0_ei = v2->estimate();
        Eigen::Isometry3d T_e_s = v3->estimate();

        Eigen::Isometry3d T_err = T_e_s * T_s0_si * T_e_s.inverse() * T_e0_ei.inverse();
        Eigen::Quaterniond q_err(T_err.rotation());
        Sophus::SO3d SO3_err(q_err);
        Eigen::Vector3d so3_err = SO3_err.log();
        // so3_err.setZero();
        Eigen::Vector3d t_err = T_err.translation();
        // t_err.setZero();

        _error << so3_err, t_err;

    }

    bool read(istream &is) override
    {
        return false;
    }

    bool write(ostream &os) const override
    {
        return false;
    }
};

int main(int argc, char** argv)
{
    if (argc != 3) {
        cout << "Usage: traj_se3_align stamped_traj_estimate.txt stamped_groundtruth.txt" << endl;
        return 1;
    }

    ifstream fin1(argv[1]);
    ifstream fin2(argv[2]);
    if (!fin1 || !fin2) {
        cout << "trajectory file does not exist!" << endl;
        return 1;
    }

    ros::init(argc, argv, "traj_se3_align");
    ros::NodeHandle nh;

    map<double, Eigen::Isometry3d> poses_sonar;
    map<double, int> time_v1id;
    map<double, Eigen::Isometry3d> poses_odom;

    std::string line;
    while (std::getline(fin1, line)) {
        if (line[0] == '#') {
            continue;
        }
        std::istringstream iss(line);
        double timestamp, tx, ty, tz, qx, qy, qz, qw;
        if (iss >> timestamp >> tx >> ty >> tz >> qx >> qy >> qz >> qw) {
            Eigen::Quaterniond q1(qw, qx, qy, qz);
            Eigen::Isometry3d T_s0_si= Eigen::Isometry3d::Identity();
            T_s0_si.rotate(q1.toRotationMatrix());
            T_s0_si.pretranslate(Eigen::Vector3d(tx, ty, tz));
            poses_sonar.insert(make_pair(timestamp, T_s0_si));
        }
    }
    fin1.close();

    while (std::getline(fin2, line)) {
        if (line[0] == '#') {
            continue;
        }
        std::istringstream iss(line);
        double timestamp, tx, ty, tz, qx, qy, qz, qw;
        if (iss >> timestamp >> tx >> ty >> tz >> qx >> qy >> qz >> qw) {
            Eigen::Quaterniond q1(qw, qx, qy, qz);
            Eigen::Isometry3d T_e0_ei = Eigen::Isometry3d::Identity();
            T_e0_ei.rotate(q1.toRotationMatrix());
            T_e0_ei.pretranslate(Eigen::Vector3d(tx, ty, tz));
            poses_odom.insert(make_pair(timestamp, T_e0_ei));
        }
    }
    fin2.close();

    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType* linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3* solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    int id = 0;
    for (auto it: poses_sonar) {
        double time = it.first;
        Eigen::Isometry3d T_s0_si = it.second;
        if (poses_odom.count(time) == 0) {
            cout << fixed << setprecision(9) << "no corresponding pose in odom: " << time << endl;
            continue;
        }
        g2o::SE3Quat est(T_s0_si.rotation(), T_s0_si.translation());
        VertexSE3Expmap* vSE3 = new VertexSE3Expmap();
        vSE3->setId(id);
        vSE3->setEstimate(est);
        vSE3->setFixed(true);
        optimizer.addVertex(vSE3);
        time_v1id.insert(make_pair(time, id));
        id++;
    }

    //set extrinsic
    Eigen::Isometry3d T_e_s = Eigen::Isometry3d::Identity();
    // Eigen::AngleAxisd a_x(M_PI, Eigen::Vector3d::UnitX());
    // T_e_s.rotate(a_x.toRotationMatrix());
    g2o::SE3Quat est_ex(T_e_s.rotation(), T_e_s.translation());
    VertexSE3Expmap* vSE3_ex = new VertexSE3Expmap();
    vSE3_ex->setId(id);
    vSE3_ex->setEstimate(est_ex);
    vSE3_ex->setFixed(false);
    optimizer.addVertex(vSE3_ex);
    id++;

    int pose_num = id;
    for (auto it: poses_sonar) {
        double time = it.first;
        if (poses_odom.count(time) == 0) {
            // cout << fixed << setprecision(9) << "no corresponding pose in odom: " << time << endl;
            continue;
        }
        Eigen::Isometry3d T_e0_ei = poses_odom[time];
        g2o::SE3Quat est(T_e0_ei.rotation(), T_e0_ei.translation());
        VertexSE3Expmap* vSE3 = new VertexSE3Expmap();
        vSE3->setId(id);
        vSE3->setEstimate(est);
        vSE3->setFixed(true);
        optimizer.addVertex(vSE3);
        id++;

        int v1id = time_v1id[time];
        EdgeSE3Align* edge = new EdgeSE3Align();
        // edge->setId(0);
        edge->setVertex(0, dynamic_cast<VertexSE3Expmap*>(optimizer.vertex(v1id)));
        edge->setVertex(1, vSE3);
        edge->setVertex(2, vSE3_ex);
        edge->setInformation(Eigen::Matrix<double, 6, 6>::Identity());
        optimizer.addEdge(edge);

        cout << fixed << setprecision(9) << "add pose: " << time << endl;

    }


    cout << "before opt T_e_s: \n" << T_e_s.matrix() << endl;
    optimizer.initializeOptimization();
    optimizer.setVerbose(true);
    optimizer.optimize(100);
    Eigen::Isometry3d T_e_s_opt = vSE3_ex->estimate();
    cout << "after opt T_e_s: \n" << T_e_s_opt.matrix() << endl;

    T_e_s.setIdentity();
    T_e_s.rotate(T_e_s_opt.rotation());
    T_e_s = T_e_s_opt;


    ofstream outfile("/home/da/project/ros/direct_sonar_ws/src/direct_sonar_odometry/evaluation/stamped_traj_estimate_opt.txt", ios::out);
    outfile << "#timestamp x y z qx qy qz qw\n";
    for (auto it: poses_sonar) {
        double time = it.first;
        Eigen::Isometry3d T_s0_si = it.second;
        Eigen::Isometry3d T_e0_ei = T_e_s * T_s0_si * T_e_s.inverse();
        Eigen::Vector3d t = T_e0_ei.translation();
        Eigen::Quaterniond q(T_e0_ei.rotation());
        outfile << fixed << setprecision(12) << time << " " << t.x() << " " << t.y() << " " << t.z()
                << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
    }

    //publish pose using MarkerArray
    visualization_msgs::MarkerArray markers_sonar_pose;
    id = 0;
    for(auto it:poses_sonar){
        Eigen::Isometry3d T_s0_si = it.second;
        Eigen::Quaterniond q_s0_si(T_s0_si.rotation());
        visualization_msgs::Marker sonar_pose_marker;
        sonar_pose_marker.header.frame_id = "map";
        sonar_pose_marker.header.stamp = ros::Time::now();
        sonar_pose_marker.ns = "sonar_pose";
        sonar_pose_marker.id = id;
        sonar_pose_marker.type = visualization_msgs::Marker::ARROW;
        sonar_pose_marker.action = visualization_msgs::Marker::ADD;
        sonar_pose_marker.pose.position.x = T_s0_si.translation().x();
        sonar_pose_marker.pose.position.y = T_s0_si.translation().y();
        sonar_pose_marker.pose.position.z = T_s0_si.translation().z();
        sonar_pose_marker.pose.orientation.x = q_s0_si.x();
        sonar_pose_marker.pose.orientation.y = q_s0_si.y();
        sonar_pose_marker.pose.orientation.z = q_s0_si.z();
        sonar_pose_marker.pose.orientation.w = q_s0_si.w();
        sonar_pose_marker.scale.x = 0.5;
        sonar_pose_marker.scale.y = 0.1;
        sonar_pose_marker.scale.z = 0.1;
        sonar_pose_marker.color.a = 0.5;
        sonar_pose_marker.color.r = 0.0;
        sonar_pose_marker.color.g = 0.0;
        sonar_pose_marker.color.b = 1.0;
        markers_sonar_pose.markers.push_back(sonar_pose_marker);
        id++;

    }

    visualization_msgs::MarkerArray markers_odom_pose;
    id = 0;
    for(auto it:poses_odom){
        double time = it.first;
        if(poses_sonar.count(time) == 0){
            continue;
        }
        Eigen::Isometry3d T_e0_ei = it.second;
        Eigen::Quaterniond q_e0_ei(T_e0_ei.rotation());
        visualization_msgs::Marker sonar_pose_marker;
        sonar_pose_marker.header.frame_id = "map";
        sonar_pose_marker.header.stamp = ros::Time::now();
        sonar_pose_marker.ns = "odom_pose";
        sonar_pose_marker.id = id;
        sonar_pose_marker.type = visualization_msgs::Marker::ARROW;
        sonar_pose_marker.action = visualization_msgs::Marker::ADD;
        sonar_pose_marker.pose.position.x = T_e0_ei.translation().x();
        sonar_pose_marker.pose.position.y = T_e0_ei.translation().y();
        sonar_pose_marker.pose.position.z = T_e0_ei.translation().z();
        sonar_pose_marker.pose.orientation.x = q_e0_ei.x();
        sonar_pose_marker.pose.orientation.y = q_e0_ei.y();
        sonar_pose_marker.pose.orientation.z = q_e0_ei.z();
        sonar_pose_marker.pose.orientation.w = q_e0_ei.w();
        sonar_pose_marker.scale.x = 0.5;
        sonar_pose_marker.scale.y = 0.1;
        sonar_pose_marker.scale.z = 0.1;
        sonar_pose_marker.color.a = 0.5;
        sonar_pose_marker.color.r = 1.0;
        sonar_pose_marker.color.g = 0.0;
        sonar_pose_marker.color.b = 0.0;
        markers_odom_pose.markers.push_back(sonar_pose_marker);
        id++;

    }

    visualization_msgs::MarkerArray markers_sonar_pose_opt;
    id = 0;
    for(auto it:poses_sonar){
        Eigen::Isometry3d T_s0_si = it.second;
        Eigen::Isometry3d T_e0_ei = T_e_s * T_s0_si * T_e_s.inverse();
        Eigen::Quaterniond q_e0_ei(T_e0_ei.rotation());
        visualization_msgs::Marker sonar_pose_marker;
        sonar_pose_marker.header.frame_id = "map";
        sonar_pose_marker.header.stamp = ros::Time::now();
        sonar_pose_marker.ns = "sonar_pose";
        sonar_pose_marker.id = id;
        sonar_pose_marker.type = visualization_msgs::Marker::ARROW;
        sonar_pose_marker.action = visualization_msgs::Marker::ADD;
        sonar_pose_marker.pose.position.x = T_e0_ei.translation().x();
        sonar_pose_marker.pose.position.y = T_e0_ei.translation().y();
        sonar_pose_marker.pose.position.z = T_e0_ei.translation().z();
        sonar_pose_marker.pose.orientation.x = q_e0_ei.x();
        sonar_pose_marker.pose.orientation.y = q_e0_ei.y();
        sonar_pose_marker.pose.orientation.z = q_e0_ei.z();
        sonar_pose_marker.pose.orientation.w = q_e0_ei.w();
        sonar_pose_marker.scale.x = 0.5;
        sonar_pose_marker.scale.y = 0.1;
        sonar_pose_marker.scale.z = 0.1;
        sonar_pose_marker.color.a = 0.5;
        sonar_pose_marker.color.r = 0.0;
        sonar_pose_marker.color.g = 1.0;
        sonar_pose_marker.color.b = 0.0;
        markers_sonar_pose_opt.markers.push_back(sonar_pose_marker);
        id++;

    }

    ros::Publisher pub_sonar_pose = nh.advertise<visualization_msgs::MarkerArray>("sonar_pose_marker", 100);
    ros::Publisher pub_odom_pose = nh.advertise<visualization_msgs::MarkerArray>("odom_pose_marker", 100);
    ros::Publisher pub_odom_pose2 = nh.advertise<visualization_msgs::MarkerArray>("sonar_pose_opt_marker", 100);
    ros::Rate loop_rate(1);
    while(ros::ok()){
        pub_sonar_pose.publish(markers_sonar_pose);
        pub_odom_pose.publish(markers_odom_pose);
        pub_odom_pose2.publish(markers_sonar_pose_opt);
        ros::spinOnce();
        loop_rate.sleep();
    }





    return 0;
}
