#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <chrono>
#include <ctime>
#include <climits>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <string>


#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

using namespace std;
using namespace g2o;


struct Measurement
{
    Measurement(Eigen::Vector3d p, double g) : pos_world(p), grayscale(g) {}

    Eigen::Vector3d pos_world;
    double grayscale;
};


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

bool
poseEstimationSonarDirect(const vector<Measurement> &meas, cv::Mat gray, Eigen::Isometry3d &Tcw,
                          double theta, double tx, double ty, double scale, cv::Mat& pre_img, cv::Mat& img);


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
        // check x,y is in the image
        if (x - 4 < 0 || (x + 4) > image_.cols || (y - 4) < 0 || (y + 4) > image_.rows) {
            _error(0, 0) = 0.0;
            this->setLevel(1);
        }
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
    // get a gray scale value from reference image (bilinear interpolated)
    inline double getPixelValue(double x, double y)
    {
        uchar* data = &image_.data[int(y) * image_.step + int(x)];
        double xx = x - floor(x);
        double yy = y - floor(y);
        return double((1 - xx) * (1 - yy) * data[0] + xx * (1 - yy) * data[1] +
                      (1 - xx) * yy * data[image_.step] + xx * yy * data[image_.step + 1]);
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


int main(int argc, char** argv)
{
    srand((unsigned int) time(0));
    string path_to_dataset = "/home/da/project/ros/direct_sonar_ws/src/direct_sonar_odometry/sonar_data";

    Eigen::Isometry3d Tcw = Eigen::Isometry3d::Identity();
    Eigen::AngleAxisd angle(0.01*M_PI,Eigen::Vector3d::UnitZ());
    // Tcw.rotate(angle.matrix());
    double range = 10;

    cv::Mat prev_color;

    vector<Measurement> measurements;
    for (int index = 1; index <= 5; index++) {
        string img_path = path_to_dataset + "/" + std::to_string(index) + ".png";
        //open image
        cv::Mat img = cv::imread(img_path);
        if (img.data == nullptr) {
            cerr << "image " << img_path << " is empty!" << endl;
            continue;
        }
        double scale = (img.rows) / range;
        double theta = -0.5 * M_PI;
        double tx = 0.5 * img.cols;
        double ty = img.rows;
        // output img opencv type
        cout << "image type: " << type2str(img.type()) << endl;
        // convert to uchar
        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        cout << "after conversion image type: " << type2str(gray.type()) << endl;


        if (index == 1) {
            // select the pixels with high gradiants
            for (int x = 10; x < gray.cols - 10; x++)
                for (int y = 10; y < gray.rows - 10; y++) {
                    Eigen::Vector2d delta(gray.ptr<uchar>(y)[x + 1] - gray.ptr<uchar>(y)[x - 1],
                                          gray.ptr<uchar>(y + 1)[x] - gray.ptr<uchar>(y - 1)[x]);
                    if (delta.norm() < 100) {
                        continue;
                    }
                    Eigen::Vector2d p_pixel(x, y);
                    Eigen::Vector3d p3d = sonar2Dto3D(p_pixel, theta, tx, ty, scale);
                    Eigen::Vector2d p2d_test = sonar3Dto2D(p3d, theta, tx, ty, scale);
                    double grayscale = double(gray.ptr<uchar>(y)[x]);
                    measurements.push_back(Measurement(p3d, grayscale));
                }
            prev_color = img.clone();
            cout << "add total " << measurements.size() << " measurements." << endl;
            continue;
        }


        cv::Mat img_show(img.rows * 2, img.cols, CV_8UC3);
        prev_color.copyTo(img_show(cv::Rect(0, 0, img.cols, img.rows)));
        img.copyTo(img_show(cv::Rect(0, img.rows, img.cols, img.rows)));
        cv::imshow("original", img_show);
        cv::waitKey(1);

        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
        poseEstimationSonarDirect(measurements, gray, Tcw, theta, tx, ty, scale,prev_color,img);
        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
        chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
        cout << "direct method costs time: " << time_used.count() << " seconds." << endl;
        cout << "Tcw=" << Tcw.matrix() << endl;


    }
    return 0;
}

bool
poseEstimationSonarDirect(const vector<Measurement> &measurements, cv::Mat gray, Eigen::Isometry3d &Tcw,
                          double theta, double tx, double ty, double scale, cv::Mat& pre_img, cv::Mat& img)
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
    optimizer.setVerbose(true);

    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
    pose->setEstimate(g2o::SE3Quat(Tcw.rotation(), Tcw.translation()));
    pose->setId(0);
    optimizer.addVertex(pose);

    // add edge
    int id = 1;
    for (Measurement m: measurements) {
        EdgeSE3SonarDirect* edge = new EdgeSE3SonarDirect(m.pos_world, theta, tx, ty, scale, gray);
        edge->setVertex(0, pose);
        edge->setMeasurement(m.grayscale);
        edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
        edge->setId(id++);
        //set robust kernel
        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
        rk->setDelta(10.0);
        edge->setRobustKernel(rk);
        optimizer.addEdge(edge);
    }
    cout << "edges in graph: " << optimizer.edges().size() << endl;

    for(int i=0;i<1;i++){
        optimizer.initializeOptimization();
        optimizer.optimize(100);
        Eigen::Isometry3d T = pose->estimate();
        Tcw = T;

        cout<<"Tcw=\n"<<Tcw.matrix()<<endl;

        // plot the feature points
        cv::Mat img_show(img.rows * 2, img.cols, CV_8UC3);
        pre_img.copyTo(img_show(cv::Rect(0, 0, img.cols, img.rows)));
        img.copyTo(img_show(cv::Rect(0, img.rows, img.cols, img.rows)));
        for (Measurement m: measurements) {
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
        cv::imshow("result", img_show);
        cv::waitKey(0);
        cv::imwrite("/home/da/project/ros/direct_sonar_ws/src/direct_sonar_odometry/results/result.png", img_show);
    }
    return true;
}