//
// Created by da on 03/08/23.
//

#ifndef SRC_OPTIMIZATIONTYPE_H
#define SRC_OPTIMIZATIONTYPE_H

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
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include "Frame.h"

using namespace std;
using namespace g2o;


class VertexSonarPose : public BaseVertex<6, Eigen::Isometry3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexSonarPose(){}
    VertexSonarPose(const Eigen::Isometry3d& T_si_s0);
    virtual bool read(std::istream &is){}
    virtual bool write(std::ostream &os) const{}
    virtual void setToOriginImpl()
    {
        _estimate.setIdentity();
    }
    virtual void oplusImpl(const double *update_);
};
class VertexSonarPoint : public BaseVertex<3,Eigen::Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexSonarPoint(){}
    VertexSonarPoint(const Eigen::Vector3d& p);
    virtual bool read(std::istream &is){}
    virtual bool write(std::ostream &os) const{}
    virtual void setToOriginImpl()
    {
        _estimate.setZero();
    }
    virtual void oplusImpl(const double *update_);
};


class EdgeSE3Sonar : public BaseBinaryEdge<2, double, VertexSonarPose, VertexSonarPoint>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeSE3Sonar() {}

    EdgeSE3Sonar(const Eigen::Vector2d &p_pixel, double theta, double tx, double ty, double scale)
            : x_pixel_(p_pixel), theta_(theta), tx_(tx), ty_(ty), scale_(scale) {}

    virtual void computeError();

    // plus in manifold
    // virtual void linearizeOplus();

    // dummy read and write functions because we don't care...
    virtual bool read(std::istream &in) {}

    virtual bool write(std::ostream &out) const {}

public:
    Eigen::Vector2d x_pixel_;
    double theta_;
    double tx_;
    double ty_;
    double scale_;

protected:
    inline Eigen::Matrix3d skew_symetric_matrix(const Eigen::Vector3d &v)
    {
        Eigen::Matrix3d S; // Skew-symmetric matrix
        S << 0, -v(2), v(1), v(2), 0, -v(0), -v(1), v(0), 0;
        return S;
    }
};

class EdgeSE3Odom : public BaseBinaryEdge<6, double, VertexSonarPose, VertexSonarPose>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeSE3Odom() {}

    EdgeSE3Odom(Eigen::Isometry3d &T_si_sj) : mT_si_sj(T_si_sj) {}

    virtual void computeError();

    // plus in manifold
    // virtual void linearizeOplus();

    // dummy read and write functions because we don't care...
    virtual bool read(std::istream &in) {}

    virtual bool write(std::ostream &out) const {}

public:
    Eigen::Isometry3d mT_si_sj;
};

class EdgeSE3SonarDirect : public BaseUnaryEdge<1, double, VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeSE3SonarDirect() {}

    EdgeSE3SonarDirect(const Eigen::Vector3d &point, double theta, double tx, double ty, double scale,
                       double range, double fov, cv::Mat image) : x_world_(point), theta_(theta),
                                                                  tx_(tx), ty_(ty), scale_(scale),
                                                                  range_(range), fov_(fov),
                                                                  image_(image.clone()) {}

    virtual void computeError();

    // plus in manifold
    virtual void linearizeOplus();

    // dummy read and write functions because we don't care...
    virtual bool read(std::istream &in) {}

    virtual bool write(std::ostream &out) const {}

protected:
    bool IsMarginalPoint(double x, double y)
    {
        Eigen::Vector3d p_3d = sonar2Dto3D(Eigen::Vector2d(x, y), theta_, tx_, ty_, scale_);
        //cartesian to polar
        double r = sqrt(p_3d.x() * p_3d.x() + p_3d.y() * p_3d.y());
        if ((r < 0.3) || (r > range_ - 0.3)) {
            return true;
        }
        else if ((theta_ < -0.5 * fov_ + 0.05) || (theta_ > 0.5 * fov_ - 0.05)) {
            return true;
        }
        else {
            return false;
        }
    }

    inline double getSinlePixelValue(double x, double y)
    {
        if (x < 0 || x >= image_.cols - 1 || y < 0 || y >= image_.rows - 1) {
            // Handle the out-of-bounds case. Here, we simply return 0, but you can adjust as needed.
            return 0.0;
        }

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
    double theta_, tx_, ty_, scale_, range_, fov_; // Camera intrinsics
    cv::Mat image_;    // reference image
};

#endif //SRC_OPTIMIZATIONTYPE_H
