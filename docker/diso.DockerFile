FROM osrf/ros:noetic-desktop-full

# RUN 
RUN apt update && apt install vim -y
SHELL ["/bin/bash", "-c"]

# setup clion remote development
RUN apt-get update \
  && apt-get install -y ssh \
      build-essential \
      gcc \
      g++ \
      gdb \
      clang \
      make \
      ninja-build \
      cmake \
      autoconf \
      automake \
      locales-all \
      dos2unix \
      rsync \
      tar \
      python \
  && apt-get clean

RUN useradd -m da \
  && yes 123 | passwd da

RUN usermod -s /bin/bash da

# install ceres Dependencies

# CMake
RUN apt-get install cmake -y
# google-glog + gflags
RUN apt-get install libgoogle-glog-dev libgflags-dev -y
# BLAS & LAPACK
RUN apt-get install libatlas-base-dev -y
# SuiteSparse and CXSparse (optional)
RUN apt-get install libsuitesparse-dev -y
# opencv boost eigen3
RUN sudo apt install libopencv-dev libboost-dev libboost-filesystem-dev libeigen3-dev -y
# ros dependencies
RUN sudo apt-get update && sudo apt install ros-noetic-pcl-ros ros-noetic-tf2-sensor-msgs ros-noetic-image-geometry ros-noetic-cv-bridge -y
# Clang ninja git
RUN sudo apt install ninja-build clang-10 lldb-10 git -y

# Build dependencies
RUN sudo apt-get install -y --no-install-recommends apt-utils
# RUN sudo apt-get install -y \
#       build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev\
#       libavformat-dev libswscale-dev python-dev python-numpy libtbb2 libtbb-dev\
#       libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev \
#       libboost-all-dev \      
#       libvtk6-dev libgtk-3-dev \      
#       libatlas-base-dev gfortran \
#       libparmetis-dev \      
#       libtool

RUN sudo apt install -y \
      build-essential cmake git pkg-config libgtk-3-dev \
      libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
      libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev \
      gfortran openexr libatlas-base-dev python3-dev python3-numpy \
      libtbb2 libtbb-dev libdc1394-22-dev      





# RUN yes 123 | passwd root
# RUN useradd -m da && yes 123 | passwd da

RUN sudo apt install ninja-build clang-10 lldb-10 -y

# install opencv
WORKDIR /third_party
RUN git clone https://github.com/opencv/opencv.git
RUN git clone https://github.com/opencv/opencv_contrib.git
WORKDIR /third_party/opencv
RUN git checkout 1ac7baceff84768d583887fe7c7f1883e9f8e0cb
WORKDIR /third_party/opencv_contrib
RUN git checkout b8eca4161439f361ae4d5a398aab1495136c704b
WORKDIR /third_party/opencv/build
RUN cmake .. -GNinja\
      -DCMAKE_CXX_COMPILER=/usr/bin/clang++-10\
      -DCMAKE_C_COMPILER=/usr/bin/clang-10\
      -DOPENCV_EXTRA_MODULES_PATH=/third_party/opencv_contrib/modules
      # -DCMAKE_INSTALL_PREFIX=/home/da/project/Thirdparty/opencv3.2
RUN ninja && ninja install


#install eigen
WORKDIR /third_party
RUN git clone https://gitlab.com/libeigen/eigen.git
WORKDIR /third_party/eigen
RUN git checkout 3dc3a0ea2d0773af4c0ffd7bbcb21c608e28fcef
WORKDIR /third_party/eigen/build
RUN cmake .. && make -j8 && make install

#install fmt
WORKDIR /third_party
RUN git clone https://github.com/fmtlib/fmt.git
WORKDIR /third_party/fmt
RUN git checkout c240d98ffd37b65ee7c27185c62a2cbc093ff185
WORKDIR /third_party/fmt/build
RUN cmake .. && make -j8 && make install

#install g2o
WORKDIR /third_party
RUN git clone https://github.com/RainerKuemmerle/g2o.git
WORKDIR /third_party/g2o
RUN git checkout 21b7ce45c9c129879f800a7047d3e4be7251444f
WORKDIR /third_party/g2o/build
RUN cmake .. -DCMAKE_BUILD_TYPE=Release && make -j8 && make install

#install pcl
WORKDIR /third_party
RUN git clone https://github.com/PointCloudLibrary/pcl.git
WORKDIR /third_party/pcl
RUN git checkout af3ce2530b7ae8ed083a3515168626c587a5bbcd
WORKDIR /third_party/pcl/build
RUN cmake .. -DCMAKE_BUILD_TYPE=Release && make -j8 && make install

#install sophus
WORKDIR /third_party
RUN git clone https://github.com/strasdat/Sophus.git
WORKDIR /third_party/Sophus
RUN git checkout 49a7e1286910019f74fb4f0bb3e213c909f8e1b7
WORKDIR /third_party/Sophus/build
RUN cmake .. && make -j8 && make install

# Install sudo
RUN apt-get update && apt-get install -y sudo ros-noetic-pcl-conversions

# # Create the user 'da' and set password '123'
# RUN useradd -m -s /bin/bash da && echo 'da:123' | chpasswd

# Give 'da' sudo privileges
RUN usermod -aG sudo da

# To avoid password prompts for sudo commands, you can set 'da' to have passwordless sudo:
RUN echo "da ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER da
WORKDIR /home/da/diso_ws/src
RUN git clone https://github.com/SenseRoboticsLab/DISO.git
WORKDIR /home/da/diso_ws/src/DISO
RUN git checkout main
WORKDIR /home/da/diso_ws
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make"
#
RUN echo "source /opt/ros/noetic/setup.bash" >> /home/da/.bashrc
RUN echo "source /home/da/diso_ws/devel/setup.bash" >> /home/da/.bashrc

CMD ["/bin/bash"]