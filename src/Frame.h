//
// Created by da on 03/08/23.
//

#ifndef SRC_FRAME_H
#define SRC_FRAME_H
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core/core.hpp>
#include <vector>
#include <map>
#include <set>
#include <memory>
#include <utility>
#include <thread>
#include <shared_mutex>
using namespace std;
class MapPoint;
class Frame
{
public:
    static int mFrameNum;
    int mID;
    void ComputePyramid(const cv::Mat &img, vector<cv::Mat> &img_pyramid_out, int layer = 5);
    void DetectKeyPoints();
    bool IsMarginalPoint(double x, double y);
    void AddObservation(shared_ptr<MapPoint> p_mp, const pair<double,double>& key);
    void RemoveObservation(int mp_id);
    void VisualizeHist(const cv::Mat mat);
    //init from sonar image
    Frame(const cv::Mat &img, double timestamp, double range, double FOV, int layer = 5, double gradient_threshold = 0.1);
    Frame(const cv::Mat &img, double timestamp, double range, double FOV, Eigen::Isometry3d T_b0_bi, int layer = 5, double gradient_threshold = 0.1);
    Frame(const Frame& f);
public:
    // sonar img
    cv::Mat mImg;
    //image pyramid
    vector<cv::Mat> mPyramid;
    int mPyramidLayer;
    //timestamp
    double mTimestamp;

    //key points
    set<pair<double,double>> mKeyPoints;
    set<pair<double,double>> mInliers;
    double mGradientThreshold;

    //Sonar Parameter
    double mRange,mTx,mTy,mTheta,mScale,mFOV;

private:
    // frame pose
    // protected by mFrameMutex
    Eigen::Isometry3d mT_s0_si;
    Eigen::Isometry3d mT_b0_bi;
public:
    const Eigen::Isometry3d &GetPose() const;
    const Eigen::Isometry3d &GetOdomPose() const;

    void SetPose(const Eigen::Isometry3d &T_w_sj);

    const map<pair<double, double>, shared_ptr<MapPoint>> &GetObservationsF2L() const;

    void SetObservationsF2L(const map<pair<double, double>, shared_ptr<MapPoint>> &ObservationsF2L);

    const map<int, pair<double, double>> &GetObservationsL2F() const;

    void SetObservationsL2F(const map<int, pair<double, double>> &ObservationsL2F);

private:
    //observation Pixel_Pos -> MapPoint pointer, Frame to Landmark
    // protected by mFrameMutex
    map<pair<double,double>, shared_ptr<MapPoint>> mObservations_F2L;
    //observation landmark ID -> Pixel_Pos, Landmark to Frame
    // protected by mFrameMutex
    map<int,pair<double,double>> mObservations_L2F;

    //mutex
    mutable shared_mutex mFrameMutex;


};
Eigen::Vector3d
sonar2Dto3D(const Eigen::Vector2d &p_pixel, double theta, double tx, double ty, double scale);
Eigen::Vector2d
sonar3Dto2D(const Eigen::Vector3d &p_sonar3D, double theta, double tx, double ty, double scale);
double getPixelValue(const cv::Mat &img, double x, double y);
double getPatternPixelValue(const cv::Mat &img, double x, double y);

#endif //SRC_FRAME_H
