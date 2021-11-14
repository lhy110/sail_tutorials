# Slam-kitti_dataset
This tutorial uses hdl slam to map kitti dataset.

#### Prerequisites
* ROS melodic or noetic
* Catkin Workspace
* RVIZ


## Downloading kitti dataset
First, navigate to the website that contains [Kitti Dataset](http://www.cvlibs.net/datasets/kitti/raw_data.php).

![image](https://user-images.githubusercontent.com/76409272/132110601-027e5612-f89d-4fcb-808c-bc26acd17945.png)

Click the log in button and register an account using your student email.
After registering, you will receive an email to verify your identity and activate your account (check you spam inbox).

After activating your account, under raw data download the 5th city drive.

![image](https://user-images.githubusercontent.com/76409272/132110692-e1c6be08-a8f2-4c5d-a4b9-82dd89ac085a.png)

Make sure to download the **synced+ rectified** dataset and the file will be in the downloads folder.

## Unzipping the dataset and converting to rosbag
We need to convert the dataset to a [rosbag](http://wiki.ros.org/rosbag) file.

Download the Kitti to bag converter.
```
pip install kitti2bag
```

To unzip the file, go to the downloads foler.
```
cd ~/Downloads/
unzip 2011_09_26_drive_0005_sync.zip
unzip 2011_09_26_calib.zip
```

Then covert the file.
```
kitti2bag -t 2011_09_26 -r 0005 raw_synced .
```

In this case, we are using the 2011_09_26_drive_0005 file but for other files, replace the dates and drive number when appropriate.

## Rosbag info
To check the rosbag contatins the right information we need, we can check the contents inside the rosbag file. [Rosbag](http://wiki.ros.org/rosbag/Commandline) contains more information about rosbag commandlines.
```
rosbag info kitti_2011_09_26_drive_0005_synced.bag
```
Then a similar result to the following will be printed out.
```
path:        kitti_2011_09_26_drive_0002_synced.bag
version:     2.0
duration:    7.8s
start:       Sep 26 2011 13:02:44.33 (1316995364.33)
end:         Sep 26 2011 13:02:52.16 (1316995372.16)
size:        417.3 MB
messages:    1078
compression: none [308/308 chunks]
types:       geometry_msgs/TwistStamped [98d34b0043a2093cf9d9345ab6eef12e]
             sensor_msgs/CameraInfo     [c9a58c1b0b154e0e6da7578cb991d214]
             sensor_msgs/Image          [060021388200f6f0f447d0fcd9c64743]
             sensor_msgs/Imu            [6a62c6daae103f4ff57a132d6f95cec2]
             sensor_msgs/NavSatFix      [2d3a8cd499b9b4a0249fb98fd05cfa48]
             sensor_msgs/PointCloud2    [1158d486dd51d683ce2f1be655c3c181]
             tf2_msgs/TFMessage         [94810edda583a504dfda3829e70d7eec]
topics:      /kitti/camera_color_left/camera_info    77 msgs    : sensor_msgs/CameraInfo    
             /kitti/camera_color_left/image_raw      77 msgs    : sensor_msgs/Image         
             /kitti/camera_color_right/camera_info   77 msgs    : sensor_msgs/CameraInfo    
             /kitti/camera_color_right/image_raw     77 msgs    : sensor_msgs/Image         
             /kitti/camera_gray_left/camera_info     77 msgs    : sensor_msgs/CameraInfo    
             /kitti/camera_gray_left/image_raw       77 msgs    : sensor_msgs/Image         
             /kitti/camera_gray_right/camera_info    77 msgs    : sensor_msgs/CameraInfo    
             /kitti/camera_gray_right/image_raw      77 msgs    : sensor_msgs/Image         
             /kitti/oxts/gps/fix                     77 msgs    : sensor_msgs/NavSatFix     
             /kitti/oxts/gps/vel                     77 msgs    : geometry_msgs/TwistStamped
             /kitti/oxts/imu                         77 msgs    : sensor_msgs/Imu           
             /kitti/velo/pointcloud                  77 msgs    : sensor_msgs/PointCloud2   
             /tf                                     77 msgs    : tf2_msgs/TFMessage        
             /tf_static                              77 msgs    : tf2_msgs/TFMessage
```

The topics shows that the bag file contains colour camera, gray camera, imu, and LiDAR data. Ope

## Downloading hdl_slam
#### Library requirements
* OpenMP
* PCL
* g2o
* suitesparse

To download OpenMP, PCL and suiteparse use the following code below:
```
sudo apt-get install libomp-dev
sudo apt-get install libpcl-dev
sudo apt-get install libsuitesparse-dev
```

To download g2o:
```
sudo apt-get install libeigen3-dev
git clone https://github.com/RainerKuemmerle/g2o.git
cd ~/g2o/
cmake ../g2o
make
sudo make install
```

#### Ros package requirements

* geodesy
* nmea_msgs
* pcl_ros
* ndt_omp
* fast_gicp

```
# for melodic
sudo apt-get install ros-melodic-geodesy ros-melodic-pcl-ros ros-melodic-nmea-msgs ros-melodic-libg2o
cd catkin_ws/src
git clone https://github.com/koide3/ndt_omp.git -b melodic
git clone https://github.com/SMRT-AIST/fast_gicp.git --recursive
git clone https://github.com/koide3/hdl_graph_slam

cd .. && catkin_make -DCMAKE_BUILD_TYPE=Release

# for noetic
sudo apt-get install ros-noetic-geodesy ros-noetic-pcl-ros ros-noetic-nmea-msgs ros-noetic-libg2o

cd catkin_ws/src
git clone https://github.com/koide3/ndt_omp.git
git clone https://github.com/SMRT-AIST/fast_gicp.git --recursive
git clone https://github.com/koide3/hdl_graph_slam

cd .. && catkin_make -DCMAKE_BUILD_TYPE=Release
```

For more information about [hdl_slam](https://github.com/koide3/hdl_graph_slam)

## Using hdl_slam
To visualise the point clouds follow the codes below: 
```
rosparam set use_sim_time true
roslaunch hdl_graph_slam hdl_graph_slam_kitti.launch
```
On a new window
```
roscd hdl_graph_slam/rviz
rviz -d hdl_graph_slam.rviz
```
Again, on another window go to the directory that contains the rosbag file and type the code below:
```
rosbag play --clock kitti_2011_09_26_drive_0005_synced.bag /kitti/velo/pointcloud:=/velodyne_points
```
## Visualising camera data on RVIZ
To visualise the camera data open RVIZ.
```
roscore
rosrun rviz rviz -f camera_color_left
```

When RVIZ opens, click the add button.

Then under 'By display type' click on 'Camera'
Camera section will appear on your displays as shown in picture above. Under where it says 'image topic' add '/kitti/camera_color_left/image_raw'.

![image](https://user-images.githubusercontent.com/76409272/132111819-26ab717c-91be-4b0f-9506-cccd1a1fc171.png)

Finally, play the bag file
```
rosbag play kitti_2011_09_26_drive_0005_synced.bag
```

Same can be done for other cameras by changing camera_color_left (in rosrun rviz rviz -f camera_color_left) to camera_XXXXX_XXXX and adding the following camera topic observed from rosbag info.

