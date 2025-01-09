//
// Created by da on 03/08/23.
//

#ifndef SRC_MAPPOINT_H
#define SRC_MAPPOINT_H

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <map>
#include <set>
#include <memory>
#include <utility>
#include <thread>
#include <shared_mutex>

using namespace std;
class Frame;
class MapPoint
{
public:
    MapPoint(const Eigen::Vector3d& p_w);
    MapPoint(const MapPoint& p);
    void AddObservation(shared_ptr<Frame> pF);
    void RemoveObservation(int frame_id);
    static int mMapPointNumber;
    int mID;

    const Eigen::Vector3d &getPosition() const;

    void setPosition(const Eigen::Vector3d &mPw);

    const map<int, shared_ptr<Frame>> &getObservations() const;

    int getObservationNum() const;


private:
    // map point position under world frame
    Eigen::Vector3d mPw;
    // observation frame_ID -> pixel_position
    map<int,shared_ptr<Frame>> mObservations;
    // how many time this landmark has been observed from frames
    int mObservationNum;

    mutable shared_mutex mMapPointMutex;

};


#endif //SRC_MAPPOINT_H
