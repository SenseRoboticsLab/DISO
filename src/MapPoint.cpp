//
// Created by da on 03/08/23.
//

#include "MapPoint.h"
#include "Frame.h"
#include <ros/ros.h>

int MapPoint::mMapPointNumber = 0;

MapPoint::MapPoint(const Eigen::Vector3d &p_w)
{
    mPw = p_w;
    mID = mMapPointNumber++;
    mObservationNum = 0;
}

MapPoint::MapPoint(const MapPoint &p)
{
    mPw = p.mPw;
    mID = p.mID;
    mObservations = p.mObservations;
    mObservationNum = p.mObservationNum;
}

void MapPoint::AddObservation(shared_ptr<Frame> pF)
{
    if (mObservations.count(pF->mID)) {
        // RemoveObservation(pF->mID);
        // unique_lock<shared_mutex> lock(mMapPointMutex);
        // mObservations.insert(make_pair(pF->mID, pF));
        // mObservationNum++;
        return;
    }
    unique_lock<shared_mutex> lock(mMapPointMutex);
    mObservations.insert(make_pair(pF->mID, pF));
    mObservationNum++;
}

void MapPoint::RemoveObservation(int frame_id)
{
    unique_lock<shared_mutex> lock(mMapPointMutex);
    if (mObservations.count(frame_id)) {
        mObservations.erase(frame_id);
        mObservationNum--;
    }
}

const Eigen::Vector3d &MapPoint::getPosition() const
{
    shared_lock<shared_mutex> lock(mMapPointMutex);
    return mPw;
}

void MapPoint::setPosition(const Eigen::Vector3d &mPw)
{
    unique_lock<shared_mutex> lock(mMapPointMutex);
    MapPoint::mPw = mPw;
}

const map<int, shared_ptr<Frame>> &MapPoint::getObservations() const
{
    shared_lock<shared_mutex> lock(mMapPointMutex);
    return mObservations;
}

int MapPoint::getObservationNum() const
{
    shared_lock<shared_mutex> lock(mMapPointMutex);
    return mObservationNum;
}

