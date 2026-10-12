# TIS Smart Car OpenCV YOLO

这是一个基于 ROS 2 的智能车视觉工程。

工程主要完成两件事：

- 发布底盘控制命令 `cmd_vel`
- 订阅里程计信息 `odom`

不包含底盘通信节点，所以如果你的小车底层还有串口、CAN、驱动板通信，这部分需要你自己的底盘节点提供。

## 环境配置

先进入工程根目录：

```bash
cd TIS-smart_car_opencv_yolo
```

推荐环境：

- Ubuntu 22.04
- ROS 2 Humble
- Python 3.10

### 1. Python 环境

推荐使用 `conda`：

```bash
conda create -n opencv python=3.10 -y
conda activate opencv
```

安装 YOLO 相关依赖：

```bash
pip install -r src/yolo_cam/yolo_cam/YOLOv8/requirements.txt
pip install typeguard empy
```

主线运行时，重点保证这里的 YOLO Python 依赖正确即可。

### 2. ROS 2 依赖

```bash
sudo apt update
sudo apt install -y \
  python3-colcon-common-extensions \
  python3-rosdep \
  ros-humble-cv-bridge
```

如果还没初始化过 `rosdep`：

```bash
sudo rosdep init
rosdep update
```

安装工程 ROS 依赖：

```bash
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y
```

如果编译或启动时报错，大概率先检查 ROS 2 依赖有没有装全。

## 编译

在工程根目录执行：

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## 配置文件

`yolo_node` 的参数文件在：

`src/yolo_cam/config/yolo_node.yaml`

可以配置：

- 模型路径 `model_path`
- 推理设备 `device`
- 置信度阈值 `confidence_threshold`
- 输入输出话题
- 是否发布检测框图像

## 启动

默认启动顺序：

1. `camera_node`
2. `vision_node`
3. `yolo_node`

直接启动：

```bash
source install/setup.bash
ros2 launch turn_on_wheeltec_robot base_serial.launch.py #自带节点启动
ros2 launch yolo_cam bringup.launch.py
```

默认就是 `map:=1`。

选择不同的 `vision_node`：

```bash
ros2 launch yolo_cam bringup.launch.py map:=1 #对应环岛地图
ros2 launch yolo_cam bringup.launch.py map:=2 #对应岔路地图
ros2 launch yolo_cam bringup.launch.py map:=3 #对应大地图
```

也可以指定自己的 YOLO 配置文件：

```bash
ros2 launch yolo_cam bringup.launch.py yolo_config:=/path/to/yolo_node.yaml
```

## 单独启动

```bash
ros2 run yolo_cam camera_node
ros2 run yolo_cam yolo_node --ros-args --params-file src/yolo_cam/config/yolo_node.yaml
ros2 run vision_node vision_node
```
