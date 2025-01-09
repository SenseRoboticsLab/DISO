# DISO: Direct Imaging Sonar Odometry
![](img/1.gif)

Welcome to the **Direct Imaging Sonar Odometry (DISO)** system repository!

DISO is an advanced sonar odometry framework developed to enhance the accuracy and reliability of underwater navigation. It estimates the relative transformation between two sonar frames by minimizing aggregated sonar intensity errors at points with high intensity gradients. Key features include:
- **Diect Sonar Optimization**: Optimizes transformations between two sonar frames by minizing the overall acoustic intensity error.
- **Multi-sensor Window Optimization**: Optimizes transformations across multiple frames for robust trajectory estimation.
- **Data Association Strategy**: Efficiently matches corresponding sonar points to enhance accuracy.
- **Acoustic Intensity Outlier Rejection**: Filters out anomalous sonar readings for improved reliability.

## Linux Installation
We recommand using our docker compose file to build DISO and reproduce experiment result.

Installation has been test on **Ubuntu 20.04** and **ROS Noetic**.

```bash
mkdir -p diso_ws/src
cd diso_ws/src
git clone git@github.com:SenseRoboticsLab/DISO_InternalShare.git
cd ./DISO_InternalShare/docker
docker build --network host -f ./diso.DockerFile -t diso ./
xhost +
docker run --rm -it -v /tmp/.X11-unix:/tmp/.X11-unix -v ~/.ssh:/home/da/.ssh -e DISPLAY -e XAUTHORITY -e NVIDIA_DRIVER_CAPABILITIES=all --name diso --network host diso
# winthin the container
git clone git@github.com:SenseRoboticsLab/DISO_InternalShare.git
cd ..
catkin_make
source ./devel/setup.bash
roslaunch direct_sonar_odometry aracati2017.launch
```
You should have a rviz window jump out as follows after these steps:
![](img/2.png)

Then open a new terminal and follow the instruction to play data and visualize DISO:
```bash
#go to src of your catkin workspace
cd diso_ws/src
git clone https://github.com/SenseRoboticsLab/Aracati2017_DISO.git
```
Then download bag file from [Aracati2017 google drive](https://drive.google.com/file/d/1dbpfd3jElTdHmnceKE5RL8hzU-BDYaW-/view?usp=sharing), then place the bag file in the **diso_ws/src/Aracati2017_DISO/bags** folder.
```bash
cd ./Aracati2017_DISO/docker
docker compose up
```
## Paper
For more information, please read our [paper](https://ieeexplore.ieee.org/document/10611064)

## Citation
```
@INPROCEEDINGS{10611064,
  author={Xu, Shida and Zhang, Kaicheng and Hong, Ziyang and Liu, Yuanchang and Wang, Sen},
  booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)}, 
  title={DISO: Direct Imaging Sonar Odometry}, 
  year={2024},
  pages={8573-8579},
  doi={10.1109/ICRA57147.2024.10611064}}

```

## License

DISO is released under a GPLv3 license. For commercial purposes of DISO , please contact: sen.wang@imperial.ac.uk