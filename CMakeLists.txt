cmake_minimum_required(VERSION 3.0.2)
project(direct_sonar_odometry)

## Compile as C++17, supported in ROS Kinetic and newer
set(CMAKE_CXX_STANDARD 17)
SET(ROS_BUILD_TYPE RelWithDebInfo)
SET(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  image_transport
  nav_msgs
  roscpp
  rospy
  sensor_msgs
  std_msgs
        tf
        image_transport
)
#set(Eigen3_DIR ${PROJECT_SOURCE_DIR}/Thirdparty/eigen3.3.4)
find_package(Eigen3 3.3 REQUIRED)
#set(OpenCV_DIR /home/da/project/Thirdparty/opencv3.2/share/OpenCV)
find_package(OpenCV 3.2 REQUIRED)
find_package(PCL REQUIRED)
find_package(fmt REQUIRED)
## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)


## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend tag for "message_generation"
##   * add a build_depend and a exec_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependency has been pulled in
##     but can be declared for certainty nonetheless:
##     * add a exec_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
# generate_messages(
#   DEPENDENCIES
#   geometry_msgs#   nav_msgs#   sensor_msgs#   std_msgs
# )

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## To declare and build dynamic reconfigure parameters within this
## package, follow these steps:
## * In the file package.xml:
##   * add a build_depend and a exec_depend tag for "dynamic_reconfigure"
## * In this file (CMakeLists.txt):
##   * add "dynamic_reconfigure" to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * uncomment the "generate_dynamic_reconfigure_options" section below
##     and list every .cfg file to be processed

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES direct_sonar_odometry
#  CATKIN_DEPENDS geometry_msgs image_transport nav_msgs roscpp rospy sensor_msgs std_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
        ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/include
        ${PROJECT_SOURCE_DIR}/include/cv_bridge_slam
        ${catkin_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
        /home/da/project/ros/direct_sonar_ws/src/direct_sonar_odometry/Thirdparty/eigen3.3.4/include/eigen3
)
link_directories(
#        ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib
        ${PROJECT_SOURCE_DIR}/src/Thirdparty/g2o/lib
        ${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
set(G2O_LIBS g2o_core g2o_types_sba g2o_types_slam3d g2o_opengl_helper g2o_solver_csparse g2o_stuff g2o_csparse_extension)


## Declare a C++ library
# add_library(${PROJECT_NAME}
#   src/${PROJECT_NAME}/direct_sonar_odometry.cpp
# )

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
# add_executable(${PROJECT_NAME}_node src/direct_sonar_odometry_node.cpp)
#add_executable(direct_sonar_odometry_node src/cv_bridge.cpp src/rgb_colors.cpp src/direct_semidense_ros.cpp
#        )
#target_link_libraries(direct_sonar_odometry_node ${catkin_LIBRARIES} ${OpenCV_LIBS} ${G2O_LIBS})

add_library(direct_sonar_odometry_lib src/cv_bridge.cpp src/rgb_colors.cpp
        src/System.cpp
        src/System.h
        src/Frame.cpp
        src/Frame.h
        src/MapPoint.cpp
        src/MapPoint.h
        src/Track.cpp
        src/Track.h
        src/OptimizationType.cpp
        src/OptimizationType.h
        src/nanoflann.hpp
        src/LocalMapping.cpp
        src/LocalMapping.h)
target_link_libraries(direct_sonar_odometry_lib ${catkin_LIBRARIES} ${OpenCV_LIBS} ${G2O_LIBS} fmt::fmt-header-only)

add_executable(aracati2017_node src/run_aracati2017.cpp)
target_link_libraries(aracati2017_node direct_sonar_odometry_lib)

#add_executable(sim_node src/run_sim.cpp)
#target_link_libraries(sim_node direct_sonar_odometry_lib)


add_executable(traj_align src/traj_se3_align.cpp)
target_link_libraries(traj_align ${catkin_LIBRARIES} ${OpenCV_LIBS} ${G2O_LIBS} fmt::fmt-header-only)

add_executable(repub_gt src/RepubGT.cpp)
target_link_libraries(repub_gt ${catkin_LIBRARIES})

#add_executable(direct_sonar_odometry_test src/direct_semidense.cpp)
#target_link_libraries(direct_sonar_odometry_test ${catkin_LIBRARIES} ${OpenCV_LIBS} ${G2O_LIBS})
#
#add_executable(pyramid_test src/pyramid_test.cpp src/nanoflann.hpp)
#target_link_libraries(pyramid_test ${catkin_LIBRARIES} ${OpenCV_LIBS} ${G2O_LIBS})

## Rename C++ executable without prefix
## The above recommended prefix causes long target names, the following renames the
## target back to the shorter version for ease of user use
## e.g. "rosrun someones_pkg node" instead of "rosrun someones_pkg someones_pkg_node"
# set_target_properties(${PROJECT_NAME}_node PROPERTIES OUTPUT_NAME node PREFIX "")

## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
# target_link_libraries(${PROJECT_NAME}_node
#   ${catkin_LIBRARIES}
# )

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# catkin_install_python(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_executables.html
# install(TARGETS ${PROJECT_NAME}_node
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark libraries for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_libraries.html
# install(TARGETS ${PROJECT_NAME}
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_direct_sonar_odometry.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
