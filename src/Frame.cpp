//
// Created by da on 03/08/23.
//

#include "Frame.h"
#include "MapPoint.h"
#include <ros/ros.h>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui/highgui.hpp>

int Frame::mFrameNum = 0;

Eigen::Vector3d
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

Eigen::Vector2d
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

double getPixelValue(const cv::Mat &img, double x, double y)
{
    if (x < 0 || x >= img.cols - 1 || y < 0 || y >= img.rows - 1) {
        // Handle the out-of-bounds case. Here, we simply return 0, but you can adjust as needed.
        return 0.0;
    }
    uchar* data = &img.data[int(y) * img.step + int(x)];
    double xx = x - floor(x);
    double yy = y - floor(y);
    return double(
            (1 - xx) * (1 - yy) * data[0] + xx * (1 - yy) * data[1] + (1 - xx) * yy * data[img.step] +
            xx * yy * data[img.step + 1]);
}

double getPatternPixelValue(const cv::Mat &img, double x, double y)
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

Frame::Frame(const cv::Mat &img, double timestamp, double range, double FOV, int layer,
             double gradient_threshold) : mTimestamp(timestamp), mPyramidLayer(layer),
                                          mGradientThreshold(gradient_threshold), mRange(range),
                                          mFOV(FOV)
{
    mT_s0_si.setIdentity();
    mID = mFrameNum++;
    mImg = img.clone();
    mScale = (mImg.rows) / mRange;
    mTheta = -0.5 * M_PI;
    mTx = 0.5 * mImg.cols;
    mTy = mImg.rows;
    ComputePyramid(mImg, mPyramid, mPyramidLayer);
    DetectKeyPoints();
}

Frame::Frame(const cv::Mat &img, double timestamp, double range, double FOV, Eigen::Isometry3d T_b0_bi,
             int layer, double gradient_threshold) : mTimestamp(timestamp), mPyramidLayer(layer),
                                                     mGradientThreshold(gradient_threshold),
                                                     mRange(range), mFOV(FOV), mT_b0_bi(T_b0_bi)
{
    mT_s0_si.setIdentity();
    mID = mFrameNum++;
    mImg = img.clone();
    mScale = (mImg.rows) / mRange;
    mTheta = -0.5 * M_PI;
    mTx = 0.5 * mImg.cols;
    mTy = mImg.rows;
    ComputePyramid(mImg, mPyramid, mPyramidLayer);
    DetectKeyPoints();
}

void Frame::ComputePyramid(const cv::Mat &img, vector<cv::Mat> &img_pyramid_out, int layer)
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


void Frame::DetectKeyPoints()
{
    // Load the image

    cv::Mat gray, results;
    cv::cvtColor(mImg, gray, cv::COLOR_BGR2GRAY);
    results = mImg.clone();
    cv::Mat grad_x, grad_y;
    cv::Mat abs_grad_x, abs_grad_y, grad;

    // Compute gradients (you can adjust the kernel size depending on your needs)
    cv::Sobel(gray, grad_x, CV_32F, 1, 0, 7);
    cv::Sobel(gray, grad_y, CV_32F, 0, 1, 7);

    // Computing the magnitude of the gradients
    cv::magnitude(grad_x, grad_y, grad);

    cv::Mat floatImage;  // Assuming this is your CV_32F image.

    // Find the minimum and maximum values in your floatImage.
    double minVal, maxVal;
    cv::minMaxLoc(grad, &minVal, &maxVal);
    // ROS_INFO_STREAM("grad max:" << maxVal << " min:" << minVal);

    // Normalize the float image to the range [-128, 127].
    cv::Mat normalized_grad;
    cv::normalize(grad, normalized_grad, 0, 255, cv::NORM_MINMAX, CV_32F);

    // Convert the datatype from CV_32F to CV_8S.
    normalized_grad.convertTo(grad, CV_8U);
    // VisualizeHist(grad);


    // Apply threshold
    cv::Mat mask;
    cv::threshold(grad, mask, mGradientThreshold, 255, cv::THRESH_BINARY);


    // Convert to CV_8U
    mask.convertTo(mask, CV_8U);


    // Reshape to 1D array and sort indices in decreasing order
    cv::Mat grad1d = grad.reshape(0, 1);
    cv::Mat sortedIndices;
    cv::sortIdx(grad1d, sortedIndices, cv::SORT_EVERY_ROW + cv::SORT_DESCENDING);

    // Number of points you want, make sure it's less than or equal to total number of pixels
    int N = std::min(1500, grad1d.cols);

    // Get the top N gradient points
    cv::Mat topNPoints = sortedIndices(cv::Range::all(), cv::Range(0, N));

    // // Draw green circles on original image at the points with top gradients
    // for (int i = 0; i < N; ++i) {
    //     int idx = topNPoints.at<int>(0, i);
    //     int x = idx / grad.cols;  // Convert index back to 2D coordinates
    //     int y = idx % grad.cols;
    //
    //     if (mask.at<unsigned char>(x, y) == 255) {
    //         if (!IsMarginalPoint(y, x)) {
    //             cv::circle(results, cv::Point(y, x), 1, cv::Scalar(0, 255, 0), -1);
    //             mKeyPoints.insert(make_pair(double(y), double(x)));
    //         }
    //     }
    // }
    double x = 4.0;
    // Define the grid size based on image size and distance x
    int gridRows = grad.rows / x;
    int gridCols = grad.cols / x;

    // Create a grid of points initialized with (-1,-1) indicating no point and their gradient values
    std::vector<std::vector<std::pair<cv::Point, float>>> grid(gridRows,
                                                               std::vector<std::pair<cv::Point, float>>(
                                                                       gridCols, {{-1, -1}, -1.0f}));


    for (int i = 0; i < N; ++i) {
        int idx = topNPoints.at<int>(0, i);
        int row = idx / grad.cols;  // Convert index back to 2D coordinates
        int col = idx % grad.cols;

        int gridRow = row / x;
        int gridCol = col / x;

        // float gradientValue = grad1d.at<float>(0, i);

        // Ensure grid indices are valid
        if (gridRow >= 0 && gridRow < gridRows && gridCol >= 0 && gridCol < gridCols) {

            float gradientValue = grad1d.at<float>(0, i);

            // Ensure mask indices are valid
            if (row >= 0 && row < mask.rows && col >= 0 && col < mask.cols) {
                if (gradientValue > grid[gridRow][gridCol].second &&
                    mask.at<unsigned char>(row, col) == 255 && !IsMarginalPoint(col, row)) {
                    grid[gridRow][gridCol] = {{col, row}, gradientValue};
                }
            }
        }

    }

    // Extract points from grid and draw them
    for (int i = 0; i < gridRows; ++i) {
        for (int j = 0; j < gridCols; ++j) {
            cv::Point pt = grid[i][j].first;
            if (pt.x != -1 && pt.y != -1) {
                cv::circle(results, pt, 1, cv::Scalar(0, 255, 0), -1);
                mKeyPoints.insert(make_pair(double(pt.x), double(pt.y)));
            }
        }
    }
    ROS_INFO_STREAM("KeyPoints: " << mKeyPoints.size() << endl);

}

bool Frame::IsMarginalPoint(double x, double y)
{
    Eigen::Vector3d p_3d = sonar2Dto3D(Eigen::Vector2d(x, y), mTheta, mTx, mTy, mScale);
    //cartesian to polar
    double r = sqrt(p_3d.x() * p_3d.x() + p_3d.y() * p_3d.y());
    double theta_ = atan2(p_3d.y(), p_3d.x());
    if ((r < 0.3) || (r > mRange - 0.3)) {
        return true;
    }
    else if ((theta_ < -0.5 * mFOV + 0.05) || (theta_ > 0.5 * mFOV - 0.05)) {
        return true;
    }
    else {
        return false;
    }
}

Frame::Frame(const Frame &f)
{
    mT_s0_si = f.mT_s0_si;
    mT_b0_bi = f.mT_b0_bi;
    mID = f.mID;
    mImg = f.mImg.clone();
    mKeyPoints = f.mKeyPoints;
    mInliers = f.mInliers;
    mTimestamp = f.mTimestamp;
    mPyramidLayer = f.mPyramidLayer;
    mGradientThreshold = f.mGradientThreshold;
    mRange = f.mRange;
    mFOV = f.mFOV;
    mScale = f.mScale;
    mTheta = f.mTheta;
    mTx = f.mTx;
    mTy = f.mTy;
    for (int i = 0; i < mPyramidLayer; i++) {
        cv::Mat p_copy = f.mPyramid[i].clone();
        mPyramid.push_back(p_copy.clone());
    }
    mObservations_F2L = f.mObservations_F2L;
    mObservations_L2F = f.mObservations_L2F;
}

const Eigen::Isometry3d &Frame::GetPose() const
{
    shared_lock<shared_mutex> lock(mFrameMutex);
    return mT_s0_si;
}

void Frame::SetPose(const Eigen::Isometry3d &T_w_sj)
{
    unique_lock<shared_mutex> lock(mFrameMutex);
    mT_s0_si = T_w_sj;
}

const map<pair<double, double>, shared_ptr<MapPoint>> &Frame::GetObservationsF2L() const
{
    shared_lock<shared_mutex> lock(mFrameMutex);
    return mObservations_F2L;
}

void Frame::SetObservationsF2L(const map<pair<double, double>, shared_ptr<MapPoint>> &ObservationsF2L)
{
    unique_lock<shared_mutex> lock(mFrameMutex);
    mObservations_F2L = ObservationsF2L;
}

const map<int, pair<double, double>> &Frame::GetObservationsL2F() const
{
    shared_lock<shared_mutex> lock(mFrameMutex);
    return mObservations_L2F;
}

void Frame::SetObservationsL2F(const map<int, pair<double, double>> &ObservationsL2F)
{
    unique_lock<shared_mutex> lock(mFrameMutex);
    mObservations_L2F = ObservationsL2F;
}

void Frame::AddObservation(shared_ptr<MapPoint> p_mp, const pair<double, double> &key)
{
    if (mObservations_L2F.count(p_mp->mID)) {
        // ROS_WARN_STREAM("Observation already exists. MapPoint" << p_mp->mID << " frame" << mID);
        // RemoveObservation(p_mp->mID);
        // unique_lock<shared_mutex> lock(mFrameMutex);
        // mObservations_L2F.insert(make_pair(p_mp->mID, key));
        // mObservations_F2L.insert(make_pair(key, p_mp));
        return;
    }
    unique_lock<shared_mutex> lock(mFrameMutex);
    mObservations_L2F.insert(make_pair(p_mp->mID, key));
    mObservations_F2L.insert(make_pair(key, p_mp));
}

void Frame::RemoveObservation(int mp_id)
{
    unique_lock<shared_mutex> lock(mFrameMutex);
    if (mObservations_L2F.count(mp_id)) {
        if (mObservations_F2L.count(mObservations_L2F[mp_id])) {
            mObservations_F2L.erase(mObservations_L2F[mp_id]);
        }
        mObservations_L2F.erase(mp_id);
    }
}

const Eigen::Isometry3d &Frame::GetOdomPose() const
{
    std::shared_lock<shared_mutex> lock(mFrameMutex);
    return mT_b0_bi;
}

void Frame::VisualizeHist(const cv::Mat mat)
{
    // Assuming mat is a single-channel image.
    int histSize = 256; // For 8-bit image, you can adjust for other types
    float range[] = {0, 256}; // For 8-bit image
    const float* histRange = {range};
    cv::Mat hist;
    cv::calcHist(&mat, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);

    int hist_w = 512;
    int hist_h = 400;
    int bin_w = cvRound((double) hist_w / histSize);
    cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));

    // Normalize the histogram to fit in the histImage.
    cv::normalize(hist, hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());

    for (int i = 1; i < histSize; i++) {
        cv::line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
                 cv::Point(bin_w * i, hist_h - cvRound(hist.at<float>(i))), cv::Scalar(255, 0, 0), 2, 8,
                 0);
    }

    int tickLength = 5;  // Length of each tick, can be adjusted
    int tickInterval = 25; // Interval between ticks, can be adjusted
    for (int i = 0; i <= histSize; i += tickInterval) {
        cv::line(histImage, cv::Point(bin_w * i, hist_h), cv::Point(bin_w * i, hist_h - tickLength),
                 cv::Scalar(255, 255, 255), 1, 8, 0);
        cv::putText(histImage, std::to_string(i), cv::Point(bin_w * i, hist_h - tickLength - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }


    cv::imshow("Histogram", histImage);
    cv::waitKey(1);


}
