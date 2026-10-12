#!/bin/bash
echo '========================================'
echo '  最终版本巡线'
echo '========================================'

# 清理之前的进程
pkill -9 -f base_serial
pkill -9 -f astra_camera
pkill -9 -f start_toggle_node
pkill -9 -f line_follower
pkill -9 -f cmd_mux
pkill -9 -f detection_turn
pkill -9 -f ros_yolo_detect
pkill -9 -f image_compressor
sleep 2

source /opt/ros/humble/setup.bash
SMARTCAR_ROOT=/home/pavs-akm/zzh_space/TIS-smart_car-main
PARAMS_FILE=$SMARTCAR_ROOT/line_follow_ws/src/line_follower/config/line_follower_params.yaml

# 1. 启动底盘
echo '[1/4] 启动底盘...'
source ~/Pav-S_ws/install/setup.bash
ros2 launch turn_on_wheeltec_robot base_serial.launch.py > /tmp/smartcar_base.log 2>&1 &
BASE_PID=$!
sleep 3

# 2. 启动相机
echo '[2/4] 启动相机...'
ros2 launch astra_camera astro_pro_plus.launch.xml > /tmp/smartcar_camera.log 2>&1 &
CAMERA_PID=$!
sleep 5

# 3. 启动YOLOv8数字识别
echo '[3/4] 启动YOLOv8数字识别'
cd $SMARTCAR_ROOT/yolov5
python3 ros_yolo_detect_v8.py --ros-args --params-file $PARAMS_FILE > /tmp/smartcar_yolo.log 2>&1 &
YOLO_PID=$!
sleep 3

# 4. 启动巡线节点
echo '[4/4] 启动巡线节点...'
source $SMARTCAR_ROOT/line_follow_ws/install/setup.bash
ros2 launch line_follower line_follower_with_yolo.launch.py params_file:=$PARAMS_FILE > /tmp/smartcar_line.log 2>&1 &
LINE_PID=$!
sleep 2

echo '========================================'
echo '  ✓ 最终版本已启动'
echo '按空格键启动/停止巡线'
echo 'Ctrl+C 退出'


# 前台运行空格键监听
ros2 run line_follower start_toggle_node

# 清理
kill $LINE_PID 2>/dev/null
kill $YOLO_PID 2>/dev/null
kill $BASE_PID 2>/dev/null
kill $CAMERA_PID 2>/dev/null
