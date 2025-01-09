#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <map>
#include <chrono>
#include <ctime>
#include <climits>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
// #include <opencv2/opencv.hpp>
#include <string>


#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <image_transport/subscriber_filter.h>


using namespace std;
using namespace g2o;


struct Measurement
{
    Measurement(Eigen::Vector3d p, double g) : pos_world(p), grayscale(g) {}

    Eigen::Vector3d pos_world;
    double grayscale;
};

inline double getPixelValue(const cv::Mat &img, double x, double y)
{
    uchar* data = &img.data[int(y) * img.step + int(x)];
    double xx = x - floor(x);
    double yy = y - floor(y);
    return double(
            (1 - xx) * (1 - yy) * data[0] + xx * (1 - yy) * data[1] + (1 - xx) * yy * data[img.step] +
            xx * yy * data[img.step + 1]);
}

// get a gray scale value from reference image (bilinear interpolated)
// {0,0}, {0,-2}, {-1,-1}, {1,-1}, {-2,0},  {2,0}, {-1,1}, {1,1}, {0,2}
inline double getPatternPixelValue(const cv::Mat &img, double x, double y)
{
    // major sample
    // {0,0}
    double x1 = x + 0;
    double y1 = y + 0;
    //minor sample
    //{0,-2}
    double x2 = x + 0;
    double y2 = y - 2;
    //{-1,-1}
    double x3 = x - 1;
    double y3 = y - 1;
    //{1,-1}
    double x4 = x + 1;
    double y4 = y - 1;
    //{-2,0}
    double x5 = x - 2;
    double y5 = y + 0;
    //{2,0}
    double x6 = x + 2;
    double y6 = y + 0;
    //{-1,1}
    double x7 = x - 1;
    double y7 = y + 1;
    //{1,1}
    double x8 = x + 1;
    double y8 = y + 1;
    //{0,2}
    double x9 = x + 0;
    double y9 = y + 2;
    // get the value of the pixel
    double v1 = getPixelValue(img, x1, y1);
    double v2 = getPixelValue(img, x2, y2);
    double v3 = getPixelValue(img, x3, y3);
    double v4 = getPixelValue(img, x4, y4);
    double v5 = getPixelValue(img, x5, y5);
    double v6 = getPixelValue(img, x6, y6);
    double v7 = getPixelValue(img, x7, y7);
    double v8 = getPixelValue(img, x8, y8);
    double v9 = getPixelValue(img, x9, y9);
    // weighted bilinear interpolation
    double value = v1 * (0.5) + (v2 + v3 + v4 + v5 + v6 + v7 + v8 + v9) * (0.5 / 8);

    return value;
}


void ComputePyramid(const cv::Mat &img, vector<cv::Mat> &img_pyramid_out, int layer = 5)
{
    img_pyramid_out.clear();
    img_pyramid_out.push_back(img.clone());
    cv::Mat src = img.clone();
    for (int i = 1; i < layer; i++) {
        cv::Mat down;
        cv::pyrDown(src, down, cv::Size(src.cols / 2, src.rows / 2));
        img_pyramid_out.push_back(down);
        src = down;
    }
}


inline Eigen::Vector3d
sonar2Dto3D(const Eigen::Vector2d &p_pixel, double theta, double tx, double ty, double scale)
{
    Eigen::Affine2d T_pixel_sonar = Eigen::Affine2d::Identity();
    Eigen::Vector2d t(tx, ty);
    Eigen::Rotation2Dd R(theta);
    T_pixel_sonar.pretranslate(t).rotate(R).scale(scale);
    Eigen::Affine2d T_sonar_pixel = T_pixel_sonar.inverse();
    Eigen::Vector2d p_sonar2D = T_sonar_pixel * p_pixel;
    Eigen::Vector3d p_sonar3D(p_sonar2D.x(), p_sonar2D.y(), 0);
    return p_sonar3D;
}

inline Eigen::Vector2d
sonar3Dto2D(const Eigen::Vector3d &p_sonar3D, double theta, double tx, double ty, double scale)
{
    Eigen::Affine2d T_pixel_sonar = Eigen::Affine2d::Identity();
    Eigen::Vector2d t(tx, ty);
    Eigen::Rotation2Dd R(theta);
    T_pixel_sonar.pretranslate(t).rotate(R).scale(scale);
    Eigen::Vector2d p_sonar2D(p_sonar3D.x(), p_sonar3D.y());
    Eigen::Vector2d p_pixel = T_pixel_sonar * p_sonar2D;

    return p_pixel;
}


bool
poseEstimationDirect(const vector<Measurement> &measurements, cv::Mat* gray, Eigen::Matrix3f &intrinsics,
                     Eigen::Isometry3d &Tcw);

bool poseEstimationSonarDirect(const vector<Measurement> &meas, cv::Mat gray, Eigen::Isometry3d &Tcw,
                               double theta, double tx, double ty, double scale, cv::Mat &pre_img,
                               cv::Mat &img);


image_transport::Publisher IMG_PUB;
ros::Publisher ODOM_PUB;
vector<Measurement> MEAS;
cv::Mat CUR_IMG;
cv::Mat PRE_IMG;
double RANGE = 50;
int FRAME_ID = 0;
int GRADIENT_THRESHOLD = 100;
int PYRAMID_LAYER = 3;
double FOV = (130.0/180.0) * M_PI;
bool IS_INITED=false;
Eigen::Isometry3d Ts0_sj = Eigen::Isometry3d::Identity();
Eigen::Isometry3d Tsj_si_pre = Eigen::Isometry3d::Identity();
Eigen::Isometry3d Tbs = Eigen::Isometry3d::Identity();
Eigen::Isometry3d T_w_s0 = Eigen::Isometry3d::Identity();
std::map<double, Eigen::Isometry3d> STAMP_POSE,STAMP_POSE_GT;

/***
*
* @param x pixel x
* @param y pixel y
* @return is the point arround polar image boundary
*/
bool IsMarginalPoint(double x, double y, double theta, double tx, double ty, double scale)
{
    Eigen::Vector3d p_3d = sonar2Dto3D(Eigen::Vector2d(x, y), theta, tx, ty, scale);
    //cartesian to polar
    double r = sqrt(p_3d.x() * p_3d.x() + p_3d.y() * p_3d.y());
    double theta_ = atan2(p_3d.y(), p_3d.x());
    if ((r < 0.1) || (r > RANGE - 0.1)) {
        return true;
    }
    else if ((theta_ < -0.5 * FOV + 0.03) || (theta_ > 0.5 * FOV - 0.03)) {
        return true;
    }
    else {
        return false;
    }
}

class EdgeSE3SonarDirect : public BaseUnaryEdge<1, double, VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeSE3SonarDirect() {}

    EdgeSE3SonarDirect(const Eigen::Vector3d &point, double theta, double tx, double ty, double scale,
                       cv::Mat image) : x_world_(point), theta_(theta), tx_(tx), ty_(ty), scale_(scale),
                                        image_(image.clone()) {}

    virtual void computeError()
    {
        const VertexSE3Expmap* v = static_cast<const VertexSE3Expmap*> ( _vertices[0] );
        Eigen::Vector3d x_local = v->estimate().map(x_world_);
        Eigen::Vector2d x_pixel = sonar3Dto2D(x_local, theta_, tx_, ty_, scale_);

        double x = x_pixel.x();
        double y = x_pixel.y();
        Eigen::Vector2d delta(getPixelValue(x+1,y)-getPixelValue(x-1,y),
                                      getPixelValue(x,y+1)-getPixelValue(x,y-1));
        // check x,y is in the image
        if (IsMarginalPoint(x, y, theta_, tx_, ty_, scale_)) {
            _error(0, 0) = 0.0;
            this->setLevel(1);
        }
        // else if (delta.norm() < GRADIENT_THRESHOLD) {
        //     _error(0, 0) = 0.0;
        //     // this->setLevel(1);
        // }
        else {
            // _measurement is the pixel intensity on the reference frame
            // getPixelValue(x, y) is the pixel intensity on the current frame, it will update
            // with current sonar pose
            _error(0, 0) = getPixelValue(x, y) - _measurement;
        }
    }

    // plus in manifold
    virtual void linearizeOplus()
    {
        if (level() == 1) {
            _jacobianOplusXi = Eigen::Matrix<double, 1, 6>::Zero();
            return;
        }
        VertexSE3Expmap* vtx = static_cast<VertexSE3Expmap*> ( _vertices[0] );
        Eigen::Isometry3d T_si_w = vtx->estimate();
        Eigen::Vector3d xyz_trans = vtx->estimate().map(x_world_);   // q in book

        Eigen::Vector2d p_pixel = sonar3Dto2D(xyz_trans, theta_, tx_, ty_, scale_);
        double u = p_pixel.x();
        double v = p_pixel.y();

        // jacobian from se3 to u,v
        // NOTE that in g2o the Lie algebra is (\omega, \epsilon), where \omega is so(3) and \epsilon the translation

        // so(3) jacobian
        Eigen::Matrix<double, 2, 2> jacobian_uv_pi;
        Eigen::Rotation2Dd R_pixel_sonar(theta_);
        jacobian_uv_pi = scale_ * R_pixel_sonar.matrix();
        Eigen::Matrix<double, 2, 3> jacobian_pi_tlocal;
        jacobian_pi_tlocal << 1, 0, 0, 0, 1, 0;
        Eigen::Matrix<double, 3, 3> jacobian_tlocal_R_si_w;
        jacobian_tlocal_R_si_w = -skew_symetric_matrix(T_si_w.rotation() * x_world_);
        Eigen::Matrix<double, 2, 3> jacobian_uv_R_si_w =
                jacobian_uv_pi * jacobian_pi_tlocal * jacobian_tlocal_R_si_w;

        Eigen::Matrix<double, 3, 3> jacobian_tlocal_t_si_si_w = Eigen::Matrix<double, 3, 3>::Identity();
        Eigen::Matrix<double, 2, 3> jacobian_uv_t_si_si_w =
                jacobian_uv_pi * jacobian_pi_tlocal * jacobian_tlocal_t_si_si_w;

        Eigen::Matrix<double, 2, 6> jacobian_uv_ksai = Eigen::Matrix<double, 2, 6>::Zero();
        jacobian_uv_ksai.block<2, 3>(0, 0) = jacobian_uv_R_si_w;
        jacobian_uv_ksai.block<2, 3>(0, 3) = jacobian_uv_t_si_si_w;


        Eigen::Matrix<double, 1, 2> jacobian_pixel_uv;

        jacobian_pixel_uv(0, 0) = (getPixelValue(u + 1, v) - getPixelValue(u - 1, v)) / 2;
        jacobian_pixel_uv(0, 1) = (getPixelValue(u, v + 1) - getPixelValue(u, v - 1)) / 2;

        Eigen::Matrix<double, 1, 6> jacobian_intensity_xi = jacobian_pixel_uv * jacobian_uv_ksai;
        _jacobianOplusXi = jacobian_pixel_uv * jacobian_uv_ksai;
    }

    // dummy read and write functions because we don't care...
    virtual bool read(std::istream &in) {}

    virtual bool write(std::ostream &out) const {}

protected:
    inline double getSinlePixelValue(double x, double y)
    {
        uchar* data = &image_.data[int(y) * image_.step + int(x)];
        double xx = x - floor(x);
        double yy = y - floor(y);
        return double((1 - xx) * (1 - yy) * data[0] + xx * (1 - yy) * data[1] +
                      (1 - xx) * yy * data[image_.step] + xx * yy * data[image_.step + 1]);
    }

    inline double getPatternPixelValue(double x, double y)
    {
        // major sample
        // {0,0}
        double x1 = x + 0;
        double y1 = y + 0;
        //minor sample
        //{0,-2}
        double x2 = x + 0;
        double y2 = y - 2;
        //{-1,-1}
        double x3 = x - 1;
        double y3 = y - 1;
        //{1,-1}
        double x4 = x + 1;
        double y4 = y - 1;
        //{-2,0}
        double x5 = x - 2;
        double y5 = y + 0;
        //{2,0}
        double x6 = x + 2;
        double y6 = y + 0;
        //{-1,1}
        double x7 = x - 1;
        double y7 = y + 1;
        //{1,1}
        double x8 = x + 1;
        double y8 = y + 1;
        //{0,2}
        double x9 = x + 0;
        double y9 = y + 2;
        // get the value of the pixel
        double v1 = getSinlePixelValue(x1, y1);
        double v2 = getSinlePixelValue(x2, y2);
        double v3 = getSinlePixelValue(x3, y3);
        double v4 = getSinlePixelValue(x4, y4);
        double v5 = getSinlePixelValue(x5, y5);
        double v6 = getSinlePixelValue(x6, y6);
        double v7 = getSinlePixelValue(x7, y7);
        double v8 = getSinlePixelValue(x8, y8);
        double v9 = getSinlePixelValue(x9, y9);
        // weighted bilinear interpolation
        double value = v1 * (0.5) + (v2 + v3 + v4 + v5 + v6 + v7 + v8 + v9) * (0.5 / 8);

        return value;
    }

    inline double getPixelValue(double x, double y)
    {
        // return getSinlePixelValue(x, y);
        return getPatternPixelValue(x, y);
    }

    inline Eigen::Matrix3d skew_symetric_matrix(const Eigen::Vector3d &v)
    {
        Eigen::Matrix3d S; // Skew-symmetric matrix
        S << 0, -v(2), v(1), v(2), 0, -v(0), -v(1), v(0), 0;
        return S;
    }

public:
    Eigen::Vector3d x_world_;   // 3D point in world frame
    double theta_, tx_, ty_, scale_; // Camera intrinsics
    cv::Mat image_;    // reference image
};

std::string type2str(int type)
{
    std::string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch (depth) {
        case CV_8U:
            r = "8U";
            break;
        case CV_8S:
            r = "8S";
            break;
        case CV_16U:
            r = "16U";
            break;
        case CV_16S:
            r = "16S";
            break;
        case CV_32S:
            r = "32S";
            break;
        case CV_32F:
            r = "32F";
            break;
        case CV_64F:
            r = "64F";
            break;
        default:
            r = "User";
            break;
    }

    r += "C";
    r += (chans + '0');

    return r;
}

bool poseEstimationSonarDirect2(cv::Mat &pre_img, cv::Mat &img, Eigen::Isometry3d &Tsj_si, double theta,
                                double tx, double ty, double scale)
{
    //convert rgb to gray
    cv::Mat pre_gray;
    cv::cvtColor(pre_img, pre_gray, cv::COLOR_BGR2GRAY);
    cv::Mat cur_gray;
    cv::cvtColor(img, cur_gray, cv::COLOR_BGR2GRAY);

    //add measurement according to GRADIENT_THRESHOLD
    vector<Measurement> measurements;
    for (int x = 0; x < pre_gray.cols; x++)
        for (int y = 0; y < pre_gray.rows; y++) {
            Eigen::Vector2d delta(pre_gray.ptr<uchar>(y)[x + 1] - pre_gray.ptr<uchar>(y)[x - 1],
                                  pre_gray.ptr<uchar>(y + 1)[x] - pre_gray.ptr<uchar>(y - 1)[x]);
            Eigen::Vector2d delta_pattern(getPatternPixelValue(pre_gray,x+1,y)-getPatternPixelValue(pre_gray,x-1,y),
                                          getPatternPixelValue(pre_gray,x,y+1)-getPatternPixelValue(pre_gray,x,y-1));
            if (delta_pattern.norm() < GRADIENT_THRESHOLD) {
                continue;
            }
            else if (IsMarginalPoint(x, y, theta, tx, ty, scale)) {
                continue;
            }
            Eigen::Vector2d p_pixel(x, y);
            Eigen::Vector3d p3d = sonar2Dto3D(p_pixel, theta, tx, ty, scale);
            Eigen::Vector2d p2d_test = sonar3Dto2D(p3d, theta, tx, ty, scale);
            // double grayscale = getPixelValue(pre_gray, x, y);
            double grayscale = getPatternPixelValue(pre_gray, x, y);
            measurements.push_back(Measurement(p3d, grayscale));
        }

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

    // add edge
    int id = 1;
    for (Measurement m: measurements) {
        EdgeSE3SonarDirect* edge = new EdgeSE3SonarDirect(m.pos_world, theta, tx, ty, scale, cur_gray);
        edge->setVertex(0, pose);
        edge->setMeasurement(m.grayscale);
        edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
        edge->setId(id++);
        //set robust kernel
        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
        rk->setDelta(10.0);
        edge->setRobustKernel(rk);
        optimizer.addEdge(edge);
    }
    cout << "edges in graph: " << optimizer.edges().size() << endl;

    for (int i = 0; i < 1; i++) {

    }
    return true;
}

void imageCallback(const sensor_msgs::ImageConstPtr &msg)
{
    try {
        FRAME_ID++;
        cv::Mat img = cv_bridge::toCvShare(msg, "bgr8")->image;
        double scale = (img.rows) / RANGE;
        double theta = -0.5 * M_PI;
        double tx = 0.5 * img.cols;
        double ty = img.rows;
        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        CUR_IMG = img.clone();

        //first frame
        if (PRE_IMG.empty()) {
            for (int x = 10; x < gray.cols - 10; x++)
                for (int y = 10; y < gray.rows - 10; y++) {
                    Eigen::Vector2d delta(gray.ptr<uchar>(y)[x + 1] - gray.ptr<uchar>(y)[x - 1],
                                          gray.ptr<uchar>(y + 1)[x] - gray.ptr<uchar>(y - 1)[x]);
                    if (delta.norm() < GRADIENT_THRESHOLD) {
                        continue;
                    }
                    Eigen::Vector2d p_pixel(x, y);
                    Eigen::Vector3d p3d = sonar2Dto3D(p_pixel, theta, tx, ty, scale);
                    Eigen::Vector2d p2d_test = sonar3Dto2D(p3d, theta, tx, ty, scale);
                    double grayscale = double(gray.ptr<uchar>(y)[x]);
                    MEAS.push_back(Measurement(p3d, grayscale));
                }
            PRE_IMG = CUR_IMG.clone();
            return;
        }

        // set initial pose
        Eigen::Isometry3d Tcw = Eigen::Isometry3d::Identity();
        // Tcw = Tcw_pre;
        poseEstimationSonarDirect(MEAS, gray, Tcw, theta, tx, ty, scale, PRE_IMG, CUR_IMG);
        // Tcw_pre = Tcw;
        //update measurement to current frame
        MEAS.clear();
        for (int x = 10; x < gray.cols - 10; x++)
            for (int y = 10; y < gray.rows - 10; y++) {
                Eigen::Vector2d delta(gray.ptr<uchar>(y)[x + 1] - gray.ptr<uchar>(y)[x - 1],
                                      gray.ptr<uchar>(y + 1)[x] - gray.ptr<uchar>(y - 1)[x]);
                if (delta.norm() < GRADIENT_THRESHOLD) {
                    continue;
                }
                Eigen::Vector2d p_pixel(x, y);
                Eigen::Vector3d p3d = sonar2Dto3D(p_pixel, theta, tx, ty, scale);
                Eigen::Vector2d p2d_test = sonar3Dto2D(p3d, theta, tx, ty, scale);
                double grayscale = double(gray.ptr<uchar>(y)[x]);
                MEAS.push_back(Measurement(p3d, grayscale));
            }
        cout << "add total " << MEAS.size() << " measurements." << endl;
        PRE_IMG = CUR_IMG.clone();

    } catch (cv_bridge::Exception &e) {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }
}

// pyramid version
void imageCallback2(const sensor_msgs::ImageConstPtr &msg)
{
    try {
        FRAME_ID++;
        cv::Mat img = cv_bridge::toCvShare(msg, "bgr8")->image;
        CUR_IMG = img.clone();

        if (!PRE_IMG.empty()) {
            vector<cv::Mat> pyramid, pre_pyramid;
            ComputePyramid(img, pyramid, PYRAMID_LAYER);
            ComputePyramid(PRE_IMG, pre_pyramid, PYRAMID_LAYER);
            Eigen::Isometry3d Tsj_si = Eigen::Isometry3d::Identity();
            Tsj_si = Tsj_si_pre;
            for (int layer = PYRAMID_LAYER - 1; layer >= 0; layer--) {
                img = pyramid[layer];
                cv::Mat pre_img = pre_pyramid[layer];
                double scale = (img.rows - 1) / RANGE;
                double theta = -0.5 * M_PI;
                double tx = 0.5 * img.cols;
                double ty = img.rows;
                poseEstimationSonarDirect2(pre_img, img, Tsj_si, theta, tx, ty, scale);
            }
            Tsj_si_pre = Tsj_si;
            Ts0_sj = Ts0_sj * Tsj_si.inverse() ;
            double time = msg->header.stamp.toSec();
            cout << fixed << setprecision(12) << time << "\n" << Ts0_sj.matrix() << endl;
            STAMP_POSE.insert(std::make_pair(time, Ts0_sj));

            //write STAMP_POSE to a local file
            std::ofstream outfile;
            outfile.open(
                    "/home/da/project/ros/direct_sonar_ws/src/direct_sonar_odometry/evaluation/stamped_traj_estimate.txt",
                    std::ios_base::out);
            outfile << "#timestamp tx ty tz qx qy qz qw" << endl;
            for (auto s_p: STAMP_POSE) {
                Eigen::Vector3d t = s_p.second.translation();
                Eigen::Quaterniond q(s_p.second.rotation());
                outfile << fixed << setprecision(12) << s_p.first << " " << t.x() << " " << t.y() << " "
                        << t.z() << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
            }
            outfile.close();


        }
        else {
            double time = msg->header.stamp.toSec();
            STAMP_POSE[time] = Ts0_sj;
        }

        PRE_IMG = CUR_IMG;


    } catch (cv_bridge::Exception &e) {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }
}

void combinedCallback(const sensor_msgs::ImageConstPtr& image_msg, const nav_msgs::Odometry::ConstPtr& odom_msg)
{
    try {
        Eigen::Vector3d t(odom_msg->pose.pose.position.x, odom_msg->pose.pose.position.y, odom_msg->pose.pose.position.z);
        Eigen::Quaterniond q(odom_msg->pose.pose.orientation.w, odom_msg->pose.pose.orientation.x,
                             odom_msg->pose.pose.orientation.y, odom_msg->pose.pose.orientation.z);
        Eigen::Isometry3d T_w_bj = Eigen::Isometry3d::Identity();
        T_w_bj.rotate(q);
        T_w_bj.pretranslate(t);

        if(T_w_s0.isApprox(Eigen::Isometry3d::Identity())){
            T_w_s0 = T_w_bj*Tbs;
        }
        Eigen::Isometry3d T_s0_sj = T_w_s0.inverse() * T_w_bj *Tbs;
        double time = odom_msg->header.stamp.toSec();
        STAMP_POSE_GT.insert(std::make_pair(time, T_s0_sj));
        // cout << fixed << setprecision(12) << time << "\n" << T_w_bj.matrix() << endl;

        Eigen::Isometry3d T_sj_si = T_s0_sj.inverse() * Ts0_sj;

        std::ofstream outfile;
        outfile.open(
                "/home/da/project/ros/direct_sonar_ws/src/direct_sonar_odometry/evaluation/stamped_groundtruth.txt",
                std::ios_base::out);
        outfile << "#timestamp tx ty tz qx qy qz qw" << endl;
        for (auto s_p: STAMP_POSE_GT) {
            Eigen::Vector3d t = s_p.second.translation();
            Eigen::Quaterniond q(s_p.second.rotation());
            outfile << fixed << setprecision(12) << s_p.first << " " << t.x() << " " << t.y() << " "
                    << t.z() << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
        }
        outfile.close();

        FRAME_ID++;
        cv::Mat img = cv_bridge::toCvShare(image_msg, "bgr8")->image;
        CUR_IMG = img.clone();

        if (!PRE_IMG.empty()) {
            vector<cv::Mat> pyramid, pre_pyramid;
            ComputePyramid(img, pyramid, PYRAMID_LAYER);
            ComputePyramid(PRE_IMG, pre_pyramid, PYRAMID_LAYER);
            Eigen::Isometry3d Tsj_si = Eigen::Isometry3d::Identity();
            Tsj_si = T_sj_si;
            for (int layer = PYRAMID_LAYER - 1; layer >= 0; layer--) {
                img = pyramid[layer];
                cv::Mat pre_img = pre_pyramid[layer];
                double scale = (img.rows - 1) / RANGE;
                double theta = -0.5 * M_PI;
                double tx = 0.5 * img.cols;
                double ty = img.rows;
                poseEstimationSonarDirect2(pre_img, img, Tsj_si, theta, tx, ty, scale);
            }
            Tsj_si_pre = Tsj_si;
            Ts0_sj = Ts0_sj * Tsj_si.inverse() ;
            double time = image_msg->header.stamp.toSec();
            cout << fixed << setprecision(12) << time << "\n" << Ts0_sj.matrix() << endl;
            STAMP_POSE.insert(std::make_pair(time, Ts0_sj));

            //write STAMP_POSE to a local file
            std::ofstream outfile;
            outfile.open(
                    "/home/da/project/ros/direct_sonar_ws/src/direct_sonar_odometry/evaluation/stamped_traj_estimate.txt",
                    std::ios_base::out);
            outfile << "#timestamp tx ty tz qx qy qz qw" << endl;
            for (auto s_p: STAMP_POSE) {
                Eigen::Vector3d t = s_p.second.translation();
                Eigen::Quaterniond q(s_p.second.rotation());
                outfile << fixed << setprecision(12) << s_p.first << " " << t.x() << " " << t.y() << " "
                        << t.z() << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
            }
            outfile.close();


        }
        else {
            double time = image_msg->header.stamp.toSec();
            STAMP_POSE[time] = Ts0_sj;
        }

        PRE_IMG = CUR_IMG;


    } catch (cv_bridge::Exception &e) {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", image_msg->encoding.c_str());
    }
}

void combinedCallback2(const sensor_msgs::ImageConstPtr& image_msg, const geometry_msgs::PoseStamped::ConstPtr& odom_msg)
{
    try {
        Eigen::Vector3d t_odom(odom_msg->pose.position.x, odom_msg->pose.position.y, odom_msg->pose.position.z);
        Eigen::Quaterniond q_odom(odom_msg->pose.orientation.w, odom_msg->pose.orientation.x,
                             odom_msg->pose.orientation.y, odom_msg->pose.orientation.z);
        Eigen::Isometry3d T_w_bj = Eigen::Isometry3d::Identity();
        T_w_bj.rotate(q_odom.toRotationMatrix());
        T_w_bj.pretranslate(t_odom);
        // T_w_bj=T_w_bj.inverse();

        if(!IS_INITED){
            T_w_s0 = T_w_bj*Tbs;
            IS_INITED = true;
        }
        Eigen::Isometry3d T_s0_sj = T_w_s0.inverse() * T_w_bj *Tbs;
        double time = odom_msg->header.stamp.toSec();
        STAMP_POSE_GT.insert(std::make_pair(time, T_s0_sj));
        // cout << fixed << setprecision(4) << time << " Ts0_sj_gt:\n" << T_s0_sj.matrix() << endl;
        // cout << fixed << setprecision(4) << time << " T_w_bj:\n" << T_w_bj.matrix() << endl;
        // cout << fixed << setprecision(4) << time << " T_w_s0:\n" << T_w_s0.matrix() << endl;
        nav_msgs::Odometry odom;

        // Fill out the Odometry message here. For example:
        odom.header.stamp = ros::Time::now();
        odom.header.frame_id = "odom";
        odom.child_frame_id = "base_link";

        // Set the position
        odom.pose.pose.position.x = T_w_bj.translation().x();
        odom.pose.pose.position.y = T_w_bj.translation().y();
        odom.pose.pose.position.z = T_w_bj.translation().z();

        q_odom = Eigen::Quaterniond(T_w_bj.rotation());
        // Set the orientation (quaternion)
        odom.pose.pose.orientation.x = q_odom.x();
        odom.pose.pose.orientation.y = q_odom.y();
        odom.pose.pose.orientation.z = q_odom.z();
        odom.pose.pose.orientation.w = q_odom.w();
        ODOM_PUB.publish(odom);

        Eigen::Isometry3d T_sj_si = T_s0_sj.inverse() * Ts0_sj;

        std::ofstream outfile;
        outfile.open(
                "/home/da/project/ros/direct_sonar_ws/src/direct_sonar_odometry/evaluation/stamped_groundtruth.txt",
                std::ios_base::out);
        outfile << "#timestamp tx ty tz qx qy qz qw" << endl;
        for (auto s_p: STAMP_POSE_GT) {
            Eigen::Vector3d t = s_p.second.translation();
            Eigen::Quaterniond q(s_p.second.rotation());
            outfile << fixed << setprecision(12) << s_p.first << " " << t.x() << " " << t.y() << " "
                    << t.z() << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
        }
        outfile.close();
        return;

        FRAME_ID++;
        cv::Mat img = cv_bridge::toCvShare(image_msg, "bgr8")->image;
        CUR_IMG = img.clone();

        if (!PRE_IMG.empty()) {
            vector<cv::Mat> pyramid, pre_pyramid;
            ComputePyramid(img, pyramid, PYRAMID_LAYER);
            ComputePyramid(PRE_IMG, pre_pyramid, PYRAMID_LAYER);
            Eigen::Isometry3d Tsj_si = Eigen::Isometry3d::Identity();
            // Tsj_si = T_sj_si;
            for (int layer = PYRAMID_LAYER - 1; layer >= 0; layer--) {
                img = pyramid[layer];
                cv::Mat pre_img = pre_pyramid[layer];
                double scale = (img.rows - 1) / RANGE;
                double theta = -0.5 * M_PI;
                double tx = 0.5 * img.cols;
                double ty = img.rows;
                poseEstimationSonarDirect2(pre_img, img, Tsj_si, theta, tx, ty, scale);
            }
            Tsj_si_pre = Tsj_si;
            Ts0_sj = Ts0_sj * Tsj_si.inverse() ;
            double time = image_msg->header.stamp.toSec();
            cout << fixed << setprecision(4) << time << " Ts0_sj:\n" << Ts0_sj.matrix() << endl;
            STAMP_POSE.insert(std::make_pair(time, Ts0_sj));

            //write STAMP_POSE to a local file
            std::ofstream outfile;
            outfile.open(
                    "/home/da/project/ros/direct_sonar_ws/src/direct_sonar_odometry/evaluation/stamped_traj_estimate.txt",
                    std::ios_base::out);
            outfile << "#timestamp tx ty tz qx qy qz qw" << endl;
            for (auto s_p: STAMP_POSE) {
                Eigen::Vector3d t = s_p.second.translation();
                Eigen::Quaterniond q(s_p.second.rotation());
                outfile << fixed << setprecision(12) << s_p.first << " " << t.x() << " " << t.y() << " "
                        << t.z() << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
            }
            outfile.close();


        }
        else {
            double time = image_msg->header.stamp.toSec();
            STAMP_POSE[time] = Ts0_sj;
        }

        PRE_IMG = CUR_IMG;


    } catch (cv_bridge::Exception &e) {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", image_msg->encoding.c_str());
    }
}

void gtCallback(const nav_msgs::OdometryConstPtr &msg)
{
    Eigen::Vector3d t(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z);
    Eigen::Quaterniond q(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x,
                         msg->pose.pose.orientation.y, msg->pose.pose.orientation.z);
    Eigen::Isometry3d T_w_bj = Eigen::Isometry3d::Identity();
    T_w_bj.rotate(q);
    T_w_bj.pretranslate(t);

    if(T_w_s0.isApprox(Eigen::Isometry3d::Identity())){
        T_w_s0 = T_w_bj*Tbs;
    }
    Eigen::Isometry3d T_s0_sj = T_w_s0.inverse() * T_w_bj *Tbs;
    double time = msg->header.stamp.toSec();
    STAMP_POSE_GT.insert(std::make_pair(time, T_s0_sj));
    // cout << fixed << setprecision(12) << time << "\n" << T_w_bj.matrix() << endl;

    std::ofstream outfile;
    outfile.open(
            "/home/da/project/ros/direct_sonar_ws/src/direct_sonar_odometry/evaluation/stamped_groundtruth.txt",
            std::ios_base::out);
    outfile << "#timestamp tx ty tz qx qy qz qw" << endl;
    for (auto s_p: STAMP_POSE_GT) {
        Eigen::Vector3d t = s_p.second.translation();
        Eigen::Quaterniond q(s_p.second.rotation());
        outfile << fixed << setprecision(12) << s_p.first << " " << t.x() << " " << t.y() << " "
                << t.z() << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
    }
    outfile.close();

}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "image_listener");
    ros::NodeHandle nh;
    // cv::namedWindow("view");
    // cv::startWindowThread();
    image_transport::ImageTransport it(nh);
    // image_transport::Subscriber sub = it.subscribe("/rexrov/blueview_p900/sonar_image", 100,
    //                                                imageCallback2);
    // image_transport::Subscriber sub = it.subscribe("/gemini/gemini/cartesain_img", 100, imageCallback2,
    //                                                image_transport::TransportHints("compressed"));
    //subcribe gt
    // ros::Subscriber sub_gt = nh.subscribe("/rexrov/pose_gt", 100, gtCallback);
    // message_filters::Subscriber<sensor_msgs::Image> image_sub(nh, "/rexrov/blueview_p900/sonar_image", 100);
    // message_filters::Subscriber<nav_msgs::Odometry> odom_sub(nh, "/rexrov/pose_gt", 1);
    // typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, nav_msgs::Odometry> MySyncPolicy;
    // message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), image_sub, odom_sub);
    // sync.registerCallback(boost::bind(&combinedCallback, _1, _2));


    image_transport::SubscriberFilter image_sub(it, "/son", 100, image_transport::TransportHints("compressed"));
    message_filters::Subscriber<geometry_msgs::PoseStamped> odom_sub(nh, "/pose_gt", 100);

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, geometry_msgs::PoseStamped> MySyncPolicy;
    message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), image_sub, odom_sub);
    sync.registerCallback(boost::bind(&combinedCallback2, _1, _2));

    IMG_PUB = it.advertise("/sonar_image_out", 1);
    ODOM_PUB = nh.advertise<nav_msgs::Odometry>("/sonar_odom", 50);

    //translate 1.15 0 0.3
    Tbs.setIdentity();
    // Tbs.pretranslate(Eigen::Vector3d(1.15, 0, 0.3));
    // Eigen::AngleAxisd a_z(0, Eigen::Vector3d::UnitZ());
    // Eigen::AngleAxisd a_y(0.3, Eigen::Vector3d::UnitY());
    // Eigen::AngleAxisd a_x(M_PI, Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd a_z(0, Eigen::Vector3d::UnitZ());
    Eigen::AngleAxisd a_y(0, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd a_x(M_PI, Eigen::Vector3d::UnitX());
    Tbs.rotate(a_x);
    Tbs.rotate(a_y);
    Tbs.rotate(a_z);

    ros::spin();
    cv::destroyWindow("view");
}

bool poseEstimationSonarDirect(const vector<Measurement> &meas, cv::Mat gray, Eigen::Isometry3d &Tcw,
                               double theta, double tx, double ty, double scale, cv::Mat &pre_img,
                               cv::Mat &img)
{
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
    pose->setEstimate(g2o::SE3Quat(Tcw.rotation(), Tcw.translation()));
    pose->setId(0);
    optimizer.addVertex(pose);

    // add edge
    int id = 1;
    for (Measurement m: meas) {
        EdgeSE3SonarDirect* edge = new EdgeSE3SonarDirect(m.pos_world, theta, tx, ty, scale, gray);
        edge->setVertex(0, pose);
        edge->setMeasurement(m.grayscale);
        edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
        edge->setId(id++);
        //set robust kernel
        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
        rk->setDelta(10.0);
        edge->setRobustKernel(rk);
        optimizer.addEdge(edge);
    }
    cout << "edges in graph: " << optimizer.edges().size() << endl;

    for (int i = 0; i < 1; i++) {
        optimizer.initializeOptimization();
        optimizer.optimize(100);
        Eigen::Isometry3d T = pose->estimate();
        Tcw = T;

        // cout << "Tcw=\n" << Tcw.matrix() << endl;
        //convert to angle axis
        // Eigen::AngleAxisd a(Tcw.rotation());
        // cout << "rotation axis: " << a.axis().transpose() << " angle: " << a.angle() << endl;

        // plot the feature points
        cv::Mat img_show(img.rows * 2, img.cols * 2, CV_8UC3);
        pre_img.copyTo(img_show(cv::Rect(0, 0, img.cols, img.rows)));
        img.copyTo(img_show(cv::Rect(0, img.rows, img.cols, img.rows)));

        pre_img.copyTo(img_show(cv::Rect(img.cols, 0, img.cols, img.rows)));
        img.copyTo(img_show(cv::Rect(img.cols, img.rows, img.cols, img.rows)));
        for (Measurement m: meas) {
            Eigen::Vector3d p = m.pos_world;
            Eigen::Vector2d pixel_prev = sonar3Dto2D(p, theta, tx, ty, scale);
            Eigen::Vector3d p2 = Tcw * m.pos_world;
            Eigen::Vector2d pixel_now = sonar3Dto2D(p2, theta, tx, ty, scale);
            if (pixel_now(0, 0) < 0 || pixel_now(0, 0) >= img.cols || pixel_now(1, 0) < 0 ||
                pixel_now(1, 0) >= img.rows) {
                continue;
            }

            float b = 0;
            float g = 250;
            float r = 0;
            img_show.ptr<uchar>(pixel_prev(1, 0))[int(pixel_prev(0, 0)) * 3] = b;
            img_show.ptr<uchar>(pixel_prev(1, 0))[int(pixel_prev(0, 0)) * 3 + 1] = g;
            img_show.ptr<uchar>(pixel_prev(1, 0))[int(pixel_prev(0, 0)) * 3 + 2] = r;

            img_show.ptr<uchar>(pixel_now(1, 0) + img.rows)[int(pixel_now(0, 0)) * 3] = b;
            img_show.ptr<uchar>(pixel_now(1, 0) + img.rows)[int(pixel_now(0, 0)) * 3 + 1] = g;
            img_show.ptr<uchar>(pixel_now(1, 0) + img.rows)[int(pixel_now(0, 0)) * 3 + 2] = r;
            cv::circle(img_show, cv::Point2d(pixel_prev(0, 0), pixel_prev(1, 0)), 2, cv::Scalar(b, g, r),
                       1);
            cv::circle(img_show, cv::Point2d(pixel_now(0, 0), pixel_now(1, 0) + img.rows), 2,
                       cv::Scalar(b, g, r), 1);
        }
        sensor_msgs::ImagePtr msg_out = cv_bridge::CvImage(std_msgs::Header(), "bgr8",
                                                           img_show).toImageMsg();
        IMG_PUB.publish(msg_out);
        //get ros time
        auto t = ros::Time::now();
        stringstream ss;
        ss << "/home/da/project/ros/direct_sonar_ws/src/direct_sonar_odometry/results/" << FRAME_ID - 1
           << "_" << FRAME_ID << ".png";
        // cv::imshow("result", img_show);
        // cv::waitKey(0);
        cv::imwrite(ss.str(), img_show);
    }
    return true;
}

