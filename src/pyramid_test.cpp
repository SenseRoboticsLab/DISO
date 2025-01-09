#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include "nanoflann.hpp"

using namespace std;
using namespace nanoflann;

// This is an exampleof a custom dataset class
struct Point2D
{
    double x, y;
};

struct PointCloud
{
    std::vector<Point2D> pts;

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const { return pts.size(); }

    // Returns the dim'th component of the idx'th point in the class:
    inline double kdtree_get_pt(const size_t idx, int dim) const
    {
        if (dim == 0) { return pts[idx].x; }
        else { return pts[idx].y; }
    }

    // Optional bounding-box computation: return false to default to a standard bbox computation loop.
    template<class BBOX>
    bool kdtree_get_bbox(BBOX &) const { return false; }
};

typedef KDTreeSingleIndexAdaptor<L2_Simple_Adaptor<double, PointCloud>, PointCloud, 2> KDTree;

int main()
{
    PointCloud cloud;

    // Fill the cloud with random points
    for (int i = 0; i < 1000; i++) {
        Point2D point;
        point.x = rand() % 1000;
        point.y = rand() % 1000;
        cloud.pts.push_back(point);
    }

    auto start = std::chrono::high_resolution_clock::now();
    //
    KDTree index(2 /*dim*/, cloud, KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
    index.buildIndex();
    auto end = std::chrono::high_resolution_clock::now();
    double time_taken = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    time_taken *= 1e-9;
    cout << "build KD tree takes " << fixed << time_taken << setprecision(9) << " sec" << endl;

    const size_t num_results = 1;
    size_t ret_index;
    double out_dist_sqr;
    nanoflann::KNNResultSet<double> resultSet(num_results);
    resultSet.init(&ret_index, &out_dist_sqr);

    for (int i = 0; i < 1000000; i++) {
        Point2D point;
        point.x = rand() % 1000;
        point.y = rand() % 1000;

        double query_pt[2] = { point.x, point.y };
        index.findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParameters(10));


    }



    end = std::chrono::high_resolution_clock::now();
    time_taken = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    time_taken *= 1e-9;
    cout << "knn search takes " << fixed << time_taken << setprecision(9) << " sec" << endl;

    std::cout << "knnSearch(nn="<<num_results<<"): \n";
    std::cout << "ret_index=" << ret_index << " out_dist_sqr=" << out_dist_sqr << endl;


    return 0;
}
