//
// Created by da on 03/08/23.
//
#include "System.h"
#include <string>
using namespace std;
int main(int argc, char **argv)
{
    ros::init(argc, argv, "direct_sonar_odometry");
    string setting = argv[1];
    System sys(setting);
    sys.runRos();

    return 0;
}
