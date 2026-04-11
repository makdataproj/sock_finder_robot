#!/bin/bash

source /opt/ros/humble/setup.bash
source ~/M3Pro_ws/install/setup.bash
source ~/yahboomcar_ws/install/setup.bash

ros2 launch M3Pro_navigation localization.launch.py \
map:=/home/jetson/M3Pro_ws/install/M3Pro_navigation/share/M3Pro_navigation/map/yahboom.yaml
