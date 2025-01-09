//
// Created by da on 03/08/23.
//

#include "OptimizationType.h"
#include <sophus/geometry.hpp>

void EdgeSE3SonarDirect::computeError()
{
    const VertexSE3Expmap* v = static_cast<const VertexSE3Expmap*> ( _vertices[0] );
    Eigen::Vector3d x_local = v->estimate().map(x_world_);
    Eigen::Vector2d x_pixel = sonar3Dto2D(x_local, theta_, tx_, ty_, scale_);

    double x = x_pixel.x();
    double y = x_pixel.y();
    // if (IsMarginalPoint(x, y)) {
    //     _error(0, 0) = 0.0;
    //     this->setLevel(1);
    //     return;
    // }

    Eigen::Vector2d delta(getPixelValue(x + 1, y) - getPixelValue(x - 1, y),
                          getPixelValue(x, y + 1) - getPixelValue(x, y - 1));

    if (delta.norm() < 20.0) {
        _error(0, 0) = 0.0;
        // this->setLevel(1);
    }
    else {
        double est = getPixelValue(x, y);
        _error(0, 0) = est - _measurement;
    }
    // check x,y is in the image
    // if (IsMarginalPoint(x, y, theta_, tx_, ty_, scale_)) {
    //     _error(0, 0) = 0.0;
    //     this->setLevel(1);
    // }
    // else if (delta.norm() < GRADIENT_THRESHOLD) {
    //     _error(0, 0) = 0.0;
    //     // this->setLevel(1);
    // }
    // else {
    // _measurement is the pixel intensity on the reference frame
    // getPixelValue(x, y) is the pixel intensity on the current frame, it will update
    // with current sonar pose

    // }

}

void EdgeSE3SonarDirect::linearizeOplus()
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

void EdgeSE3Sonar::computeError()
{
    const VertexSonarPose* v0 = static_cast<const VertexSonarPose*> ( _vertices[0] );
    const VertexSonarPoint* v1 = static_cast<const VertexSonarPoint*> ( _vertices[1] );
    Eigen::Vector3d p_world = v1->estimate();
    Eigen::Isometry3d T_si_w = v0->estimate().inverse();
    Eigen::Vector3d x_local = T_si_w * p_world;
    Eigen::Vector2d x_pixel_est = sonar3Dto2D(x_local, theta_, tx_, ty_, scale_);

    double x = x_pixel_est.x();
    double y = x_pixel_est.y();
    // Eigen::Vector2d delta(getPixelValue(x+1,y)-getPixelValue(x-1,y),
    //                       getPixelValue(x,y+1)-getPixelValue(x,y-1));
    // check x,y is in the image
    // if (IsMarginalPoint(x, y, theta_, tx_, ty_, scale_)) {
    //     _error(0, 0) = 0.0;
    //     this->setLevel(1);
    // }
    // else if (delta.norm() < GRADIENT_THRESHOLD) {
    //     _error(0, 0) = 0.0;
    //     // this->setLevel(1);
    // }
    // else {
    // _measurement is the pixel intensity on the reference frame
    // getPixelValue(x, y) is the pixel intensity on the current frame, it will update
    // with current sonar pose

    // }
    Eigen::Vector2d err = x_pixel_ - x_pixel_est;
    _error << err;
}

// void EdgeSE3Sonar::linearizeOplus()
// {
//     if (level() == 1) {
//         _jacobianOplusXi = Eigen::Matrix<double, 2, 6>::Zero();
//         return;
//     }
//     VertexSE3Expmap* v0 = static_cast<VertexSE3Expmap*> ( _vertices[0] );
//     VertexSBAPointXYZ* v1 = static_cast<VertexSBAPointXYZ*> ( _vertices[1] );
//     Eigen::Vector3d x_world = v1->estimate();
//     Eigen::Isometry3d T_si_w = v0->estimate();
//     Eigen::Vector3d xyz_trans = T_si_w * v1->estimate();   // q in book
//
//
//     // jacobian from se3 to u,v
//     // NOTE that in g2o the Lie algebra is (\omega, \epsilon), where \omega is so(3) and \epsilon the translation
//
//     // so(3) jacobian
//     Eigen::Matrix<double, 2, 2> jacobian_uv_pi;
//     Eigen::Rotation2Dd R_pixel_sonar(theta_);
//     jacobian_uv_pi = scale_ * R_pixel_sonar.matrix();
//     Eigen::Matrix<double, 2, 3> jacobian_pi_tlocal;
//     jacobian_pi_tlocal << 1, 0, 0, 0, 1, 0;
//     Eigen::Matrix<double, 3, 3> jacobian_tlocal_R_si_w;
//     jacobian_tlocal_R_si_w = -skew_symetric_matrix(T_si_w.rotation() * x_world);
//     Eigen::Matrix<double, 2, 3> jacobian_uv_R_si_w =
//             jacobian_uv_pi * jacobian_pi_tlocal * jacobian_tlocal_R_si_w;
//
//     Eigen::Matrix<double, 3, 3> jacobian_tlocal_t_si_si_w = Eigen::Matrix<double, 3, 3>::Identity();
//     Eigen::Matrix<double, 2, 3> jacobian_uv_t_si_si_w =
//             jacobian_uv_pi * jacobian_pi_tlocal * jacobian_tlocal_t_si_si_w;
//
//     Eigen::Matrix<double, 2, 6> jacobian_uv_ksai = Eigen::Matrix<double, 2, 6>::Zero();
//     jacobian_uv_ksai.block<2, 3>(0, 0) = jacobian_uv_R_si_w;
//     jacobian_uv_ksai.block<2, 3>(0, 3) = jacobian_uv_t_si_si_w;
//
//
//
//     _jacobianOplusXi = jacobian_uv_ksai;
// }
void EdgeSE3Odom::computeError()
{
    const VertexSonarPose* v0 = static_cast<const VertexSonarPose*> ( _vertices[0] );
    const VertexSonarPose* v1 = static_cast<const VertexSonarPose*> ( _vertices[1] );

    Eigen::Isometry3d T_si_s0 = v0->estimate().inverse();
    Eigen::Isometry3d T_sj_s0 = v1->estimate().inverse();
    Eigen::Isometry3d T_s0_si = T_si_s0.inverse();
    Eigen::Isometry3d T_s0_sj = T_sj_s0.inverse();
    Eigen::Isometry3d T_si_sj_est = T_si_s0 * T_s0_sj;
    Eigen::Isometry3d T_err = T_si_sj_est * mT_si_sj.inverse();
    Sophus::SO3d SO3_err(T_err.rotation());
    Eigen::Vector3d so3_err = SO3_err.log();
    // so3_err(0) = 0;
    // so3_err(1) = 0;
    Eigen::Vector3d t_err = T_err.translation();
    // t_err(1) = 0;
    // t_err(2) = 0;
    _error << so3_err, t_err;

}

VertexSonarPose::VertexSonarPose(const Eigen::Isometry3d &T_si_s0)
{
    _estimate = T_si_s0;
}

void VertexSonarPose::oplusImpl(const double* update_)
{
    // update_ 3 rotation 3 translation
    Eigen::Matrix<double, 3, 1> update_r;
    Eigen::Matrix<double, 3, 1> update_t;
    update_r.setZero();
    update_t.setZero();
    update_r << 0, 0, update_[2];
    update_t << update_[3], update_[4], 0;
    Sophus::SO3d update_SO3 = Sophus::SO3d::exp(update_r);

    Eigen::Isometry3d T_s0_si = _estimate;
    Eigen::Vector3d t_s0_si = T_s0_si.translation();
    Eigen::Matrix3d R_s0_si = T_s0_si.rotation();
    t_s0_si += R_s0_si*update_t;
    R_s0_si = R_s0_si * update_SO3.matrix();

    T_s0_si.setIdentity();
    T_s0_si.rotate(R_s0_si);
    T_s0_si.pretranslate(t_s0_si);
    _estimate = T_s0_si;


}

VertexSonarPoint::VertexSonarPoint(const Eigen::Vector3d &p)
{
    _estimate = p;
}

void VertexSonarPoint::oplusImpl(const double* update_)
{
    double x = update_[0], y = update_[1], z = update_[2];
    _estimate(0) += x;
    _estimate(1) += y;
    // _estimate(2) += z;
}
