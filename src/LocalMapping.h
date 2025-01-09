//
// Created by da on 09/08/23.
//

#ifndef SRC_LOCALMAPPING_H
#define SRC_LOCALMAPPING_H
#include <memory>
#include <utility>
#include <map>
#include <set>
#include <deque>
#include <shared_mutex>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ros/ros.h>
#include <std_srvs/Empty.h>

using namespace std;
class Frame;

class MapPoint;

class Track;

class LocalMapping;

// class LocalMapState{
// protected:
//     shared_ptr<LocalMapping> mpLocalMaper;
// public:
//     LocalMapState(shared_ptr<LocalMapping> pLocalMaper):mpLocalMaper(pLocalMaper){};
//     virtual void Update() = 0;
// };
//
// class LocalMapToUpdate : public LocalMapState{
// public:
//     LocalMapToUpdate(shared_ptr<LocalMapping> pLocalMaper):LocalMapState(pLocalMaper){};
//     void Update() override;
// };
// class LocalMapUpToDate : public LocalMapState{
// public:
//     LocalMapUpToDate(shared_ptr<LocalMapping> pLocalMaper):LocalMapState(pLocalMaper){};
//     void Update() override;
// };

class LocalMapping : public enable_shared_from_this<LocalMapping>
{
public:
    LocalMapping(shared_ptr<Track> pTracker);
    LocalMapping(shared_ptr<Track> pTracker, Eigen::Isometry3d T_bw_b0);
    void Run();
    void NotifyTracker();
    // void SetState(shared_ptr<LocalMapState> pState);
    void InsertKeyFrame(shared_ptr<Frame> pF);
    void ProcessKeyFrame();
    void OptimizeWindow();
    void PoseEstimationWindow2Frame(shared_ptr<Frame> pF_pre, shared_ptr<Frame> pF,
                                    set<pair<double, double>> &inliers_last,
                                    set<pair<double, double>> &inliers_current,
                                    map<pair<double, double>, pair<double, double>> &association, double range, double fov);
    void Visualize();
    shared_ptr<Frame> GetLastFrameInWindow();
    deque<shared_ptr<Frame>> GetWindow();
    void Reset();
    bool SaveMapCallback(std_srvs::EmptyRequest &req, std_srvs::EmptyResponse &res);
    void PubMap();


protected:
    shared_ptr<Track> mpTracker;
    // shared_ptr<LocalMapState> mState;

    // protected by mWindowMutex
    shared_mutex mWindowMutex;
    int mWindowSize = 10;
    deque<shared_ptr<Frame>> mActiveFrameWindow;

    //Frame to process, protected by mProcessingQueueMutex
    shared_mutex mProcessingQueueMutex;
    deque<shared_ptr<Frame>> mProcessingQueue;

    ros::Publisher mMarkerPub;
    ros::Publisher mPointCloudPub;
    std::map<int,shared_ptr<MapPoint>> mActiveMapPoints;
    //ros save service
    ros::ServiceServer mSaveService;
public:
    Eigen::Isometry3d mT_bw_b0;

};


#endif //SRC_LOCALMAPPING_H
