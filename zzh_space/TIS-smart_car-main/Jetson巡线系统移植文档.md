# Jetson Orin 巡线系统移植文档

## ⚠️ 重要：移植前必读

**本系统的启动脚本和文档中的所有路径都是绝对路径，在其他机器上使用时必须修改！**

### 需要修改的路径

本系统的所有路径都基于：`/home/pavs-akm/Desktop/smartcar`

**移植到其他机器时，需要全局替换为你的实际路径，例如：**
- `/home/your-username/your-workspace/smartcar`
- 或使用符号链接指向你的路径

**涉及文件：**
1. `start_final.sh` - 启动脚本中的所有路径
2. `view_debug_compressed.sh` - 调试脚本中的路径
3. 本文档中的所有示例命令

**快速替换方法：**
```bash
# 在你的smartcar目录下运行
sed -i 's|/home/pavs-akm/Desktop/smartcar|你的实际路径|g' start_final.sh
sed -i 's|/home/pavs-akm/Desktop/smartcar|你的实际路径|g' view_debug_compressed.sh
```

---

## 一、系统概述

本系统从地平线机器人移植到 Jetson Orin 平台，基于 ROS2 Humble，实现以下功能：
- 白色巡线、黄线停止、红色避障
- YOLOv8 数字识别与转向

## 二、关键文件位置

```
/home/pavs-akm/Desktop/smartcar/
├── start_final.sh              # 主启动脚本
├── view_debug_compressed.sh    # 查看调试图
├── line_follow_ws/             # 巡线工作空间
│   ├── src/line_follower/
│   │   ├── config/
│   │   │   └── line_follower_params.yaml  # 参数配置文件 ⭐
│   │   └── src/
│   │       ├── line_follower_node.cpp      # 巡线节点
│   │       └── detection_turn_controller.cpp # 转向控制器
│   └── install/
└── yolov5/                     # YOLOv8 检测
    ├── best_yolov8.pt          # 模型文件（22MB，16类：1-8, D1-D8）
    └── ros_yolo_detect_v8.py   # YOLO检测节点
```

## 三、启动步骤

### 3.1 完整启动（推荐）

```bash
cd /home/pavs-akm/Desktop/smartcar
bash start_final.sh
```

启动后会显示：
```
========================================
  最终版本巡线
========================================
[1/4] 启动底盘...
[2/4] 启动相机...
[3/4] 启动YOLOv8数字识别...
[4/4] 启动巡线节点...
========================================
  ✓ 最终版本已启动
按空格键启动/停止巡线
Ctrl+C 退出
```

### 3.2 分步启动（调试用）

如果需要单独启动某个组件：

```bash
# 1. 启动底盘
source /opt/ros/humble/setup.bash
ros2 launch turn_on_wheeltec_robot base_serial.launch.py

# 2. 启动相机
ros2 launch astra_camera astro_pro_plus.launch.xml

# 3. 启动YOLOv8数字识别
cd /home/pavs-akm/Desktop/smartcar/line_follow_ws
source install/setup.bash
python3 /home/pavs-akm/Desktop/smartcar/yolov5/ros_yolo_detect_v8.py \
  --ros-args \
  -p image_topic:=/camera/color/image_raw \
  -p debug_image:=true \
  -p use_gpu:=true

# 4. 启动空格键控制节点
ros2 run line_follower start_toggle_node

# 5. 启动巡线节点
ros2 run line_follower line_follower_node \
  --ros-args \
  --params-file /home/pavs-akm/Desktop/smartcar/line_follow_ws/install/line_follower/share/line_follower/config/line_follower_params.yaml
```

### 3.3 启动控制

- **空格键**：启动/停止巡线
- **Ctrl+C**：完全退出系统

## 四、查看调试图

### 4.1 启动调试图

```bash
cd /home/pavs-akm/Desktop/smartcar
bash view_debug_compressed.sh
```

或者手动启动：

```bash
source /opt/ros/humble/setup.bash
rqt_image_view
```

然后在界面中选择：
- `/line_follower/debug_image` - 巡线调试图像（显示检测到的线、质心、检测框）

### 4.2 调试图内容说明

调试图会显示：
检测到的黄色线
线的质心位置
线的边界框

## 五、参数调优

### 5.1 参数文件位置

```bash
/home/pavs-akm/Desktop/smartcar/line_follow_ws/install/line_follower/share/line_follower/config/line_follower_params.yaml
```

**修改参数后无需重新编译**

### 5.2 巡线参数（最常用）

```yaml
line_follower_node:
  ros__parameters:
    # 基础速度参数
    target_speed: 0.35          # 前进速度 (m/s)
    max_yaw_rate: 1.8           # 最大转向角速度 (rad/s)
    kp: 2.3                     # 比例增益，转向灵敏度

    # 左转增强
    left_turn_ratio: 1.15       # 左转时的kp倍数

    # 感兴趣区域（ROI）
    roi_top_ratio: 0.3          # 看线的位置 (0.0=最近, 1.0=最远)
                                # 值越小看得越近，值越大看得越远
```



### 5.3 颜色检测参数

```yaml
line_follower_node:
  ros__parameters:
    # 黄色检测（HSV）
    yellow_h_min: 15            # 色调最小值 (0-180)
    yellow_h_max: 35            # 色调最大值
    yellow_s_min: 80            # 饱和度最小值 (0-255)
    yellow_s_max: 255           # 饱和度最大值
    yellow_v_min: 100           # 亮度最小值 (0-255)
    yellow_v_max: 255           # 亮度最大值

    # 白色检测（HSV）
    white_h_min: 0              # 白色色调范围很宽
    white_h_max: 180
    white_s_min: 25             # ⚠️ 饱和度上限，避免误识别浅蓝
    white_s_max: 255
    white_v_min: 120            # ⚠️ 亮度下限，太低会把白色当黑色
    white_v_max: 255
```

**颜色参数调整：**

| 问题 | 调整方向 |
|------|----------|
| 黄色线检测不到 | 放宽 yellow_h_min/max, yellow_s_min, yellow_v_min |
| 黄色线误检测太多 | 收紧 yellow_h_min/max, 增大 yellow_s_min |
| 把浅蓝色当成白色 | 减小 white_s_max (当前25) |
| 白色线检测不到（太严格） | 增大 white_v_min (当前120) |

### 5.4 YOLO检测参数

修改文件：`/home/pavs-akm/Desktop/smartcar/yolov5/ros_yolo_detect_v8.py`

```python
# 第62行
results = self.model(rgb_image, verbose=False, conf=0.15, iou=0.45)
#                                              ↑置信度阈值
```


### 5.5 数字转向参数

```yaml
detection_turn_controller:
  ros__parameters:
    # y方向对齐容差
    y_align_tolerance: 0.05     

    # 检测窗口（累积多少帧后触发转向）
    frame_window: 15            # 15帧内累积检测

    # 转向参数
    turn_duration: 0.8          # 转向持续时间 (秒)
    turn_linear: 0.2            # 转向时前进速度
    yaw_gain: 1.5               # 转向增益

    # 转向冷却时间
    turn_cooldown: 2.0          # 转向后等待时间（秒）
```


```

## 六、参数修改与生效

### 6.1 修改巡线参数

```bash
# 1. 编辑参数文件
nano /home/pavs-akm/Desktop/smartcar/line_follow_ws/install/line_follower/share/line_follower/config/line_follower_params.yaml



# 2. 重启系统或节点
# 按 Ctrl+C 退出，然后重新运行
bash /home/pavs-akm/Desktop/smartcar/start_final.sh
```

### 6.2 修改YOLO参数

```bash
# 1. 编辑代码
nano /home/pavs-akm/Desktop/smartcar/yolov5/ros_yolo_detect_v8.py

# 2. 重启YOLO节点
pkill -f ros_yolo_detect_v8.py

# 3. 重新启动（或者在start_final.sh中会自动重启）
```

## 七、常用调试命令

### 7.1 查看节点状态

```bash
# 查看所有运行中的节点
ros2 node list

# 查看节点信息
ros2 node info /line_follower_node
```

### 7.2 查看话题和数据

```bash
# 查看所有话题
ros2 topic list

# 查看质心位置（巡线状态）
ros2 topic echo /line_follower/centroid

# 查看YOLO检测结果
ros2 topic echo /hobot_dnn_detection

# 查看话题发布频率
ros2 topic hz /line_follower/centroid
ros2 topic hz /hobot_dnn_detection
```

### 7.3 查看相机状态

```bash
# 查看相机话题
ros2 topic info /camera/color/image_raw

# 查看相机帧率
ros2 topic hz /camera/color/image_raw
```

##快速参考卡

### 启动
```bash
cd /home/pavs-akm/Desktop/smartcar
bash start_final.sh
# 然后按空格键启动巡线
```

### 查看调试图
```bash
bash view_debug_compressed.sh
# 选择 /line_follower/debug_image/compressed
```

### 修改参数
1. 编辑 `line_follower_params.yaml`
2. 重启系统

### 查看质心（判断转向）
```bash
ros2 topic echo /line_follower/centroid

```

### 查看YOLO检测
```bash
ros2 topic echo /hobot_dnn_detection
# class_id: 检测到的数字（1-8, D1-D8）
# score: 置信度
# x, y: 位置坐标
```

---

**文档版本：** v1.0
**最后更新：** 2026-01-19
**适用平台：** Jetson Orin + ROS2 Humble
