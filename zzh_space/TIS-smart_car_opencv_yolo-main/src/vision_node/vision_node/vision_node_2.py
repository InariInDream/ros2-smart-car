#!/usr/bin/env python3
import rclpy
import cv2
import numpy as np
import time
import ctypes
import math
from rclpy.node import Node
from sensor_msgs.msg import Imu, Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from yolo_msg.msg import Detection,Detections
from cv_bridge import CvBridge

#结束标志，检测到end_distance距离以后的黄线视为终点
total_distance = 0.0

#➕➕➕➕➕➕➕➕里程计从模式1下处理完黄线 强拐完成了，再开始计里程➕➕➕➕➕➕➕➕➕➕➕➕
mode1_yellow_straight = 0.26
mode1_yellow_turn = 0.86
#从处理完黄线开始计算
TurnRight_start_1 = 0.7     
TurnRight_end_1 = 1.33
TurnRight_start_2 = 1.67   
TurnRight_end_2 = 2.06
#从强拐进入S弯后开始计算
TurnRight_start_final =  3.3   
TurnRight_end_final = 3.7

TurnRight_start_C1 = 0.3
TurnRight_end_C1 = 1.3
TurnRight_start_C2 = 2.2
TurnRight_end_C2 = 2.65

# 图像处理+八领域变量
road_width_mode1 = 125     # 道路宽度（像素）
road_width_mode2 = 115
road_width_mode3 = 127

#道路偏差
bias_mode1 = 0
bias_mode2 = 11
bias_mode3 = 0


line_thickness = 2    # 绘制车道线宽度
edge_search_width = 90  # 起点搜索仅限距离边缘 20 像素范围内

#图2模式1和模式3一共有几个黄线
mode1_yellow_final_count = 1 
mode3_yellow_final_count = 1


select_distance_turn1 = 0.072#数字选择
select_distance_turn2 = 0.09
distance_turn1 = 0.072
distance_turn2 = 0.09
select_distance = 0.14#里程计0.43
select_turn = 0.8 #强拐角度z
num_speed = 0.7 #识别到数字速度比例系数


#servo_pd
error_p = -0.009
Kp = 0.9
Kd = 0.4#0.4

Kp_c = 1.7
Kd_c = 0.5
pid_add = 1.2
error_limit = 0.6

#area
red_area_limit = 7000.0
cone_area_limit = 10000.0
yellow_area_limit = 2100 #1800
num_area_limit = 100

# 控制参数
speed_x = 0.2  # 基础前进速度

# speed_x_red = 0.15

yellow_line_count=0

# =============================
# 全局颜色范围参数（HSV）
# 黑色区域
LOWER_BLACK = np.array([0, 0, 0])
UPPER_BLACK = np.array([180, 255, 60])

# 蓝色区域
LOWER_BLUE = np.array([90, 60, 0])
UPPER_BLUE = np.array([130, 255, 255])

#红色区域
LOWER_RED1 = np.array([0, 100, 100])
UPPER_RED1 = np.array([10, 255, 255])
LOWER_RED2 = np.array([160, 100, 100])
UPPER_RED2 = np.array([180, 255, 255])

# 黄色区域
LOWER_YELLOW = np.array([20, 80, 30])
UPPER_YELLOW = np.array([40, 255, 255])
vis_yellow = 0 
yellow_area = 0
yellow_flag = 1

# =============================
# init
# 道路线数组
left_lane = []  # 左车道线
right_lane = []  # 右车道线
center_line = []  # 中线
#yolo坐标
yolo_num_boxes = []


# 模式定义
current_mode = 1

# 检测定义
cone_detected = 0
red_detected = 0
yellow_detected_count = 0
left_num = 0
right_num = 0

line_flag=1
error = 0.0
num_detected = 0
num_area = 0

# 数字检测增强
left_num_count = 0  # 左侧数字检测计数
right_num_count = 0  # 右侧数字检测计数
num_decision_made = False  # 是否已做出数字决策

# 红色障碍物标志
right_flag = 0
left_flag = 0

frame_count = 0  # 总帧数计数器
current_x = 0



class ImageProcessor:
    @staticmethod
    def image_process(origin_img):
        """
        处理车道线图像：提取黑色车道线并返回二值图
        """
        hsv = cv2.cvtColor(origin_img, cv2.COLOR_BGR2HSV)

        # 检测黑色车道线，红色标志
        binary = cv2.inRange(hsv, LOWER_BLACK, UPPER_BLACK)
        mask_blue = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)
        mask_red1 = cv2.inRange(hsv, LOWER_RED1, UPPER_RED1)
        mask_red2 = cv2.inRange(hsv, LOWER_RED2, UPPER_RED2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        # 将“黑色车道线”反转为黑色，其余为白色
        binary = cv2.bitwise_not(binary)
        # 去除蓝色背景
        binary[mask_blue > 0] = 0  # 蓝色区域置为黑色
        binary[mask_red > 0] = 0

        if yolo_num_boxes is not None:
            for (x1, y1, x2, y2) in yolo_num_boxes:
                # 映射 YOLO 坐标到 line_img 局部坐标
                local_y1 = max(0, int(y1 - 75))
                local_y2 = min(origin_img.shape[0], int(y2 - 75))
                local_x1 = max(0, int(x1))
                local_x2 = min(origin_img.shape[1], int(x2))

                # 检查是否与 [75,140) 区域重叠
                if local_y2 > local_y1:
                    # 将检测区域设为白色
                    binary[local_y1:local_y2, local_x1:local_x2] = 255

        # 去噪（开闭操作）
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # 检查白色区域占比（如果太小则返回空图）
        white_pixels = np.sum(binary > 0)
        if white_pixels < 100:  # 可以根据图像大小调整
            return np.zeros_like(binary)

        return binary

    
    @staticmethod
    def yellow_detect(origin_img):
        global LOWER_YELLOW, UPPER_YELLOW

        hsv = cv2.cvtColor(origin_img, cv2.COLOR_BGR2HSV)

        # 黄色区域掩码
        mask_yellow = cv2.inRange(hsv, LOWER_YELLOW, UPPER_YELLOW)

        # 计算黄色像素个数
        yellow_pixels = int(np.sum(mask_yellow > 0))

        # 可视化：画轮廓
        vis = origin_img.copy()
        contours, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (0, 255, 255), 2)

        return vis, yellow_pixels

class LineProcessor:
    @staticmethod
    def find_lane_start(binary_image, side="left", min_white_run=5):
        h, w = binary_image.shape
        if side == "left":
            x_start, x_end = 0, edge_search_width
        else:
            x_start, x_end = w - edge_search_width, w - 1

        for y in range(h-1, -1, -1):
            row = binary_image[y, :]
            if side == "left":
                for x in range(x_start, x_end - min_white_run + 1):
                    if int(row[x]) == 0 and np.all(row[x+1:x+1+min_white_run] == 255):
                        return (y, x+1)
            else:
                for x in range(x_end, x_start + min_white_run - 1, -1):
                    if int(row[x]) == 0 and np.all(row[x-min_white_run:x] == 255):
                        return (y, x-1)
        return None

    @staticmethod
    def trace_line(binary_image, start_point, side="left"):
        h, w = binary_image.shape
        if start_point is None:
            return []

        visited = np.zeros_like(binary_image, dtype=bool)
        directions = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                      (1, 0), (1, -1), (0, -1), (-1, -1)]
        points = []
        q = [start_point]

        while q:
            y, x = q.pop(0)
            if not (0 <= y < h and 0 <= x < w) or visited[y, x]:
                continue
            visited[y, x] = True
            if int(binary_image[y, x]) != 255:
                continue

            # 检测白像素边界
            if side == "left" and x > 0 and binary_image[y, x - 1] == 0:
                points.append((x, y))
            elif side == "right" and x < w - 1 and binary_image[y, x + 1] == 0:
                points.append((x, y))

            # 扩展八邻域
            for dy, dx in directions:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                    if int(binary_image[ny, nx]) == 255:
                        q.append((ny, nx))

        # 每一行只保留一个点
        filtered = {}
        for x, y in points:
            if y not in filtered:
                filtered[y] = x
            else:
                if side == "left":
                    filtered[y] = min(filtered[y], x)
                else:
                    filtered[y] = max(filtered[y], x)

        points = [(filtered[y], y) for y in sorted(filtered.keys(), reverse=True)]
        return points

    @staticmethod
    def line_detect(binary_image):
        global left_lane, right_lane, center_line, road_width

        h, w = binary_image.shape
        vis = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

        left_start = LineProcessor.find_lane_start(binary_image, "left")
        right_start = LineProcessor.find_lane_start(binary_image, "right")

        left_pts = LineProcessor.trace_line(binary_image, left_start, side="left")
        right_pts = LineProcessor.trace_line(binary_image, right_start, side="right")

        left_lane.clear()
        right_lane.clear()
        center_line.clear()
   
        if current_mode == 1 :
            width = road_width_mode1
        elif current_mode == 2 :
            width = road_width_mode2
        elif current_mode == 3 :
            width = road_width_mode3
            

        if left_pts and right_pts:
            min_len = min(len(left_pts), len(right_pts))
            for i in range(min_len):
                lx, ly = left_pts[i]
                rx, ry = right_pts[i]
                cx = (lx + rx)//2
                cy = (ly + ry)//2
                center_line.append((cx, cy))
        elif left_pts:
            for x, y in left_pts:
                center_line.append((min(x + width, w-1), y))
        elif right_pts:
            for x, y in right_pts:
                center_line.append((max(x - width, 0), y))
        else:
            center_line = [(w//2, yy) for yy in range(h-1, -1, -6)]

        def draw_line(img, pts, color):
            if len(pts) >= 2:
                cv2.polylines(img, [np.array(pts, np.int32)], False, color, thickness=line_thickness)
            elif len(pts) == 1:
                x, y = pts[0]
                cv2.circle(img, (x, y), line_thickness+2, color, -1)

        draw_line(vis, left_pts, (0,0,255))
        draw_line(vis, right_pts, (0,255,0))
        draw_line(vis, center_line, (255,0,0))

        if len(center_line) >= 3:
            bottom_pt = center_line[0]  # 最下端
            mid_pt = center_line[len(center_line) // 2]  # 中间点
            top_pt = center_line[-1]  # 最上端

            # 绘制紫色原点 (BGR = (255, 0, 255))
            cv2.circle(vis, bottom_pt, 5, (255, 0, 255), -1)
            cv2.circle(vis, mid_pt, 5, (255, 0, 255), -1)
            cv2.circle(vis, top_pt, 5, (255, 0, 255), -1)

        return vis, left_pts, right_pts, center_line

class Servo_PD:
    def __init__(self, servo_p, servo_d):
        self.mode = None
        self.Kp = servo_p
        self.Kd = servo_d
        self.output_limit = 0.9
        self.last_error = 0.0
    
    def set_mode(self, mode):
        self.mode = mode

    def calc_servo_pd(self):
        global error, road_width, center_line
        top_x = center_line[-1][0]
        mid_x = center_line[len(center_line) // 2][0]
        bottom_x = center_line[0][0]

        if current_mode == 1 :
            bias = bias_mode1
        elif current_mode == 2 :
            bias = bias_mode2
        elif current_mode == 3 :
            bias = bias_mode3

        if self.mode == 1:
            if center_line:
                error = (0.9 * top_x + 0.02 * mid_x + 0.05 * bottom_x - 120) * error_p

        elif self.mode == 2:#循左线
            if center_line:
                error = (0.9 * top_x + 0.02 * mid_x + 0.05 * bottom_x - 120- bias) * error_p

        elif self.mode == 3:#循右线
            if center_line:
                error = (0.9 * top_x + 0.02 * mid_x + 0.05 * bottom_x - 120 + bias) * error_p
    
        # print(f"error:{error}")
        if error < error_limit:
            Kp = self.Kp
            Kd = self.Kd
        else:
            Kp = self.Kp * pid_add
            Kd = self.Kd

        if current_mode ==3:
            Kp = Kp_c
            Kd = Kd_c


        output = Kp * error + Kd * (error - self.last_error)  
        self.last_error = error    
        
        if output > self.output_limit:
            output = self.output_limit
        elif output < -self.output_limit:
            output = -self.output_limit
        return output

class NumSelector:
    def __init__(self, cmd_pub):
        self.cmd_pub = cmd_pub
        self.current_mode = None
        self.target_lane = None  # 目标车道（'left' or 'right'）
        self.start_x = 0  # 记录开始时的X坐标
        self.distance_travelled = 0  # 距离控制
        self.stage = 0  # 0:强拐, 1:巡线轻拐, 2:反向归正, 3:巡线归正

        self.pd_mode1 = Servo_PD(servo_p=Kp, servo_d=Kd)#循中线
        self.pd_mode1.set_mode(1)
        self.pd_mode2 = Servo_PD(servo_p=Kp, servo_d=Kd)#循左线
        self.pd_mode2.set_mode(2)
        self.pd_mode3 = Servo_PD(servo_p=Kp, servo_d=Kd)#循右线
        self.pd_mode3.set_mode(3)

    def set_target_lane(self, left_final_num, right_final_num):
        global select_distance_turn1,select_distance_turn2
        if left_final_num != 0 and right_final_num != 0:
            if left_final_num >= right_final_num:
                self.target_lane = 'left'
                print (f"error:{error}")
                # if error < -0.08 :
                #     select_distance_turn2=distance_turn2+0.2
                # else:
                #     select_distance_turn2=distance_turn2
                # if error > 0.08 :
                #     select_distance_turn1=distance_turn1+0.2
                # else:
                #     select_distance_turn1=distance_turn1

                print(f"左走: left_final_num={left_final_num} >= right_final_num={right_final_num}")
            else:
                self.target_lane = 'right'
                print (f"error:{error}")
                # if error > 0.08 :
                #     select_distance_turn2=distance_turn2+0.2
                # else:
                #     select_distance_turn2=distance_turn2
                # if error < -0.08 :
                #     select_distance_turn1=distance_turn1+0.2
                # else:
                #     select_distance_turn1=distance_turn1
                print(f"右走: right_final_num={right_final_num} > left_final_num={left_final_num}")
        
        # 初始化阶段
        self.stage = 0  # 从第一阶段开始
        self.start_x = total_distance  # 等待 update() 记录开始位置
        self.distance_travelled = 0

    def update(self):
        global total_distance

        if self.target_lane is None:
            return True   # 没有目标，直接退出

        # 当前行驶距离
        self.distance_travelled = abs(total_distance - self.start_x)
        
        # 阶段切换逻辑
        if self.stage == 0 and self.distance_travelled > select_distance_turn1:
            self.stage = 1  # 切换到巡线轻拐阶段
            print(f"进入{self.target_lane}巡线轻拐阶段")
            self.start_x = total_distance  # 重置起始距离
            self.distance_travelled = 0
            
        elif self.stage == 1 and self.distance_travelled > select_distance:
            self.stage = 2  # 切换到反向归正阶段
            print(f"进入{self.target_lane}反向归正阶段")
            self.start_x = total_distance  # 重置起始距离
            self.distance_travelled = 0
            
        elif self.stage == 2 and self.distance_travelled > select_distance_turn2:
            self.stage = 3  # 切换到巡线归正阶段
            print(f"进入{self.target_lane}巡线归正阶段")
            self.start_x = total_distance  # 重置起始距离
            self.distance_travelled = 0
            
        elif self.stage == 3 and self.distance_travelled > select_distance:
            print("数字切道完成，退出模式")
            self.start_x = 0
            self.target_lane = None
            self.stage = 0
            return True

        return False

    def control_motion(self):
        twist = Twist()

        # 阶段0：强拐阶段
        if self.stage == 0:
            if self.target_lane == 'left':
                print("左走强拐阶段")
                twist.linear.x = speed_x * num_speed
                twist.angular.z = select_turn
            else:  # right
                print("右走强拐阶段")
                twist.linear.x = speed_x * num_speed
                twist.angular.z = -select_turn

        # 阶段1：巡线轻拐阶段
        elif self.stage == 1:
            if self.target_lane == 'left':
                print("左走巡左线轻拐阶段")
                current_output = self.pd_mode2.calc_servo_pd()  # 循左线
                twist.linear.x = speed_x * num_speed + 0.03
                twist.angular.z = current_output
            else:  # right
                print("右走巡右线轻拐阶段")
                current_output = self.pd_mode3.calc_servo_pd()  # 循右线
                twist.linear.x = speed_x * num_speed + 0.03
                twist.angular.z = current_output

        # 阶段2：反向归正阶段
        elif self.stage == 2:
            if self.target_lane == 'left':
                print("左走右转归正阶段")
                twist.linear.x = speed_x * num_speed
                twist.angular.z = -select_turn  # 右转归正
            else:  # right
                print("右走左转归正阶段")
                twist.linear.x = speed_x * num_speed
                twist.angular.z = select_turn  # 左转归正

        # 阶段3：巡线归正阶段
        elif self.stage == 3:
            if self.target_lane == 'left':
                print("左走巡左线归正阶段")
                current_output = self.pd_mode2.calc_servo_pd()  # 循左线
                twist.linear.x = speed_x * num_speed + 0.03
                twist.angular.z = current_output
            else:  # right
                print("右走巡右线归正阶段")
                current_output = self.pd_mode3.calc_servo_pd()  # 循右线
                twist.linear.x = speed_x * num_speed + 0.03
                twist.angular.z = current_output

        self.cmd_pub.publish(twist)

class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
        self.get_logger().info("VisionNode started!")
        # 创建图像桥
        self.bridge = CvBridge()
        # 创建ROS订阅者
        self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(Detections, '/yolo/detections', self.yolo_callback, 10)
        # 创建ROS发布者
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.NumSelector = NumSelector(self.cmd_pub)
        # self.Red = Red(self.cmd_pub)  # 初始化Red对象
        # 创建定时器（30Hz）
        self.timer = self.create_timer(1/30.0, self.timer_callback)
 
        self.yellow_processing = False
        self.yellow_start_frame = 0

        self.stat_left = {}
        self.stat_right = {}
        self.left_final_num = None
        self.right_final_num = None

        #数字处理状态
        self.num_processing = False
        self.red_processing = False
        self.num_start_frame = 0
        
        # 第一个黄线处理
        self.mode1_yellow_start = 0.0
        self.yellow_distance = 0.0
        
        self.last_frame = None  # 图像缓存

        self.last_odom_x = None
        self.last_odom_y = None
        self.total_distance = 0.0
        

        # 初始化PD控制器
        self.pd_mode1 = Servo_PD(servo_p=Kp, servo_d=Kd)#循中线
        self.pd_mode1.set_mode(1)
        self.pd_mode2 = Servo_PD(servo_p=Kp, servo_d=Kd)#循左线
        self.pd_mode2.set_mode(2)
        self.pd_mode3 = Servo_PD(servo_p=Kp, servo_d=Kd)#循右线
        self.pd_mode3.set_mode(3)

    def yolo_callback(self, msg: Detections):
        global left_num, right_num
        global left_num_count, right_num_count, num_detected, cone_detected, red_detected
        global yolo_num_boxes
        yolo_num_boxes = []

        # 临时存储本次检测到的数字
        temp_left_num = 0
        temp_right_num = 0
        local_num_detected = 0
        local_cone_detected = 0
        local_red_detected = 0

        """接收 YOLO 检测结果"""
        for det in msg.detections:
            value = 0
            ls_name = det.cls          # 类别名称
            conf = det.conf            # 置信度
            x1, y1, x2, y2 = det.xyxy  # 边界框坐标

            if ls_name in ("1", "2", "3", "4", "5", "6", "7", "8", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"):
                num_area = abs((x2-x1)*(y2-y1))
                #记录坐标
                yolo_num_boxes.append((int(x1), int(y1), int(x2), int(y2)))

                if num_area > num_area_limit:
                    # print(f"Class: {det.cls}, Confidence: {det.conf:.2f}, num_area: {num_area}, 位置:{(x1+x2)/2}")
                    local_num_detected = 1

                    # 解析数字值
                    if ls_name == "1" or ls_name == "s1":
                        value = 1
                    elif ls_name == "2" or ls_name == "s2":
                        value = 2
                    elif ls_name == "3" or ls_name == "s3":
                        value = 3
                    elif ls_name == "4" or ls_name == "s4":
                        value = 4
                    elif ls_name == "5" or ls_name == "s5":
                        value = 5
                    elif ls_name == "6" or ls_name == "s6":
                        value = 6
                    elif ls_name == "7" or ls_name == "s7":
                        value = 7
                    elif ls_name == "8" or ls_name == "s8":
                        value = 8

                    # 根据位置判断左右
                    if (x1+x2)/2 <= 120:
                        temp_left_num = value
                    if (x1+x2)/2 > 120:
                        temp_right_num = value

        # 数字检测稳定性处理：需要连续检测3次才确认
        if local_num_detected:
            # 左侧数字检测
            if temp_left_num > 0:
                if left_num == temp_left_num or left_num == 0:
                    left_num_count += 1
                    if left_num_count >= 3:  # 连续检测3次
                        left_num = temp_left_num
                        #print(f"左侧数字确认: {left_num}")
                else:
                    # 检测到不同的数字，重置计数
                    left_num_count = 1

            # 右侧数字检测
            if temp_right_num > 0:
                if right_num == temp_right_num or right_num == 0:
                    right_num_count += 1
                    if right_num_count >= 3:  # 连续检测3次
                        right_num = temp_right_num
                        #print(f"右侧数字确认: {right_num}")
                else:
                    # 检测到不同的数字，重置计数
                    right_num_count = 1
        else:
            # 没有检测到数字，递减计数（逐渐失效）
            if left_num_count > 0:
                left_num_count -= 1
            if right_num_count > 0:
                right_num_count -= 1

        # 更新全局检测标志
        num_detected = local_num_detected
        cone_detected = local_cone_detected
        red_detected = local_red_detected

    def imu_callback(self, msg: Imu):
        ax = msg.linear_acceleration.x
        ay = msg.linear_acceleration.y
        az = msg.linear_acceleration.z
        self.get_logger().debug(f"IMU Accel: {ax:.2f}, {ay:.2f}, {az:.2f}")

    def odom_callback(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        if self.last_odom_x is not None:
            dx = x - self.last_odom_x
            dy = y - self.last_odom_y
            self.total_distance += math.sqrt(dx*dx + dy*dy)
            global total_distance
            total_distance = self.total_distance
            print(f"距离：{total_distance:.2f},,,({x:.2f},{y:.2f})")

        self.last_odom_x = x
        self.last_odom_y = y


    def publish_control_command(self):
        global current_mode, current_x, yellow_detected_count, yellow_line_count, speed_x, frame_count, num_detected
        global left_num, right_num, left_num_count, right_num_count, num_decision_made
        global left_flag, right_flag, total_distance
        
        twist = Twist()
        
        if current_mode == 1:
        # 先检查黄线（最高优先级）      
            if yellow_detected_count > 0 and not self.yellow_processing:
                self.yellow_processing = True
                self.yellow_start_frame = frame_count
                yellow_detected_count = 0
                yellow_line_count += 1
                print(f"检测到黄线")

            if self.yellow_processing:
                elapsed_frames = frame_count - self.yellow_start_frame
                self.yellow_distance = total_distance - 2.95
                
                if elapsed_frames < 8:
                    # 前8帧停止
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0
                    self.cmd_pub.publish(twist)
                    print(f"处理黄线中，帧数{elapsed_frames}")
                    return
                
                elif yellow_line_count >= mode1_yellow_final_count:
                    if self.yellow_distance < mode1_yellow_straight:  
                        twist.linear.x = speed_x
                        twist.angular.z = 0.0
                        self.cmd_pub.publish(twist)
                        print(f"黄线后直行阶段，距离: {self.yellow_distance:.3f}")
                        return
                    elif self.yellow_distance < mode1_yellow_turn:    
                        twist.linear.x = speed_x
                        twist.angular.z = -0.9
                        self.cmd_pub.publish(twist)
                        print(f"黄线后右转阶段，距离: {self.yellow_distance:.3f}")
                        return
                    else:
                        # 黄线处理完成，恢复正常巡线
                        current_mode = 2
                        self.yellow_processing = False
                        total_distance = 0.0  # 在这里清零里程计
                        self.total_distance = 0.0
                        left_num = 0
                        right_num = 0
                        print("黄线处理完成，进入正常巡线")
                        return  
                        
                else:
                    # 不是最终黄线，直接退出处理
                    self.yellow_processing = False
            # 数字处理
            if not self.num_processing and (left_num > 0 and right_num > 0):
                # 第一次进入统计，记录开始帧
                if self.num_start_frame <= 0:  
                    self.num_start_frame = frame_count
                    print(f"开始数字统计，起始帧: {self.num_start_frame}")
                elapsed_frames = frame_count - self.num_start_frame
                if not self.num_processing:
                    if elapsed_frames <=3:
                        # 左数字统计
                        self.stat_left[left_num] = self.stat_left.get(left_num, 0) + 1
                        # 右数字统计
                        self.stat_right[right_num] = self.stat_right.get(right_num, 0) + 1
                        print(f"统计中: frame={elapsed_frames}, left_num={left_num}, right_num={right_num}")

                    else:
                        self.num_processing = True

                        # 选频率最高的左数字
                        self.left_final_num = max(self.stat_left, key=self.stat_left.get)
                        # 选频率最高的右数字
                        self.right_final_num = max(self.stat_right, key=self.stat_right.get)
                        print(f"统计完成 → 最终左数字: {self.left_final_num}, 最终右数字: {self.right_final_num}")

                        # 清空统计数据以便下一次使用
                        self.stat_left.clear()
                        self.stat_right.clear()
                        self.NumSelector.set_target_lane(self.left_final_num, self.right_final_num)
                        self.num_start_frame =0
                current_output = self.pd_mode1.calc_servo_pd()
                twist.linear.x = speed_x*0.5
                twist.angular.z = current_output
                self.cmd_pub.publish(twist)
                return

            if self.num_processing:
                if not self.NumSelector.update():
                    self.NumSelector.control_motion()
                else:
                    self.num_processing = False
                    left_num = 0
                    right_num = 0
                    left_num_count = 0
                    right_num_count = 0
                    self.left_final_num = None
                    self.right_final_num = None
                    print("数字处理完成")
                return  # 数字处理期间由NumSelector发布命令

            # 正常巡线（既无黄线处理也无数字处理）
            if not self.yellow_processing and not self.num_processing:   #如果没有黄线也没有数字，就正常巡线。
                current_output = self.pd_mode1.calc_servo_pd()
                twist.linear.x = speed_x 
                twist.angular.z = current_output

        elif current_mode == 2:
            print(f"111mode2!!!!!")
            #里程计从模式1下处理完黄线 强拐完成了，再开始计里程！！
            #离开直道右拐
            
            if yellow_detected_count > 0 and not self.yellow_processing:
                self.yellow_processing = True
                self.yellow_start_frame = frame_count
                yellow_detected_count = 0
                yellow_line_count += 1
                print(f"111检测到黄线")

            if self.yellow_processing:
                elapsed_frames = frame_count - self.yellow_start_frame
                self.yellow_distance = total_distance - 2.95
                if elapsed_frames < 8:
                    # 前8帧停止
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0
                    self.cmd_pub.publish(twist)
                    print(f"111处理黄线中，帧数{elapsed_frames}")
                    return
                else:
                    self.yellow_processing = False
                    print(f"处理黄线完成，恢复巡线")
                    # 可选：发布一次恢复速度的命令（确保小车立即启动）
                    current_output = self.pd_mode1.calc_servo_pd()
                    twist.linear.x = speed_x 
                    twist.angular.z = current_output
                    self.cmd_pub.publish(twist)
                    return

            if total_distance >= TurnRight_start_1 and total_distance <= TurnRight_end_1 and self.last_odom_y <= -1.085:  #在TurnRight_start这个距离开始强行右转
                twist.linear.x = speed_x
                twist.angular.z = -0.9
                self.cmd_pub.publish(twist)
                print(f"怪出直道")
                return 
            #拐入S弯的第一个左拐   
            elif total_distance >= TurnRight_start_2 and total_distance <= TurnRight_end_2:  #在TurnRight_start这个距离开始强行左转
                twist.linear.x = speed_x
                twist.angular.z = 0.9
                self.cmd_pub.publish(twist)
                print(f"怪进S")        
                return  
            elif total_distance > TurnRight_end_2:
                current_mode = 3
                self.total_distance = 0.0    #里程计归零，便于S弯内设置强拐！！！
                self.cmd_pub.publish(twist)
                yellow_line_count=0     #黄线个数清零
                return

            if not self.yellow_processing :    
                current_output = self.pd_mode1.calc_servo_pd()
                twist.linear.x = speed_x 
                twist.angular.z = current_output
                self.cmd_pub.publish(twist)
                return

            
        elif current_mode == 3:
            if total_distance >= TurnRight_start_final and total_distance <= TurnRight_end_final:  #在TurnRight_start这个距离开始强行右转
                print("g1")
                twist.linear.x = speed_x
                twist.angular.z = -0.9
                self.cmd_pub.publish(twist)
                return    
            if total_distance >= TurnRight_start_C1 and total_distance <= TurnRight_end_C1:  #在TurnRight_start这个距离开始强行左转
                print("g2")
                twist.linear.x = speed_x
                twist.angular.z = -0.6
                self.cmd_pub.publish(twist)
                return

            if total_distance >= TurnRight_start_C2 and total_distance <= TurnRight_end_C2:  #在TurnRight_start这个距离开始强行左转
                print("g3")
                twist.linear.x = speed_x
                twist.angular.z = 0.9
                self.cmd_pub.publish(twist)
                return

            if yellow_detected_count > 0 and not self.yellow_processing:
                    self.yellow_processing = True
                    self.yellow_start_frame = frame_count
                    yellow_detected_count = 0
                    yellow_line_count +=1
                    print(f"检测到黄线")

                # 黄线处理中（最高优先级）
            if self.yellow_processing:
                elapsed_frames = frame_count - self.yellow_start_frame
                if elapsed_frames < 8:
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0
                    self.cmd_pub.publish(twist)
                    print(f"处理黄线中，帧数{elapsed_frames},left_num={left_num} : right_num={right_num}")
                    return  # 黄线处理期间不处理数字
                #终止条件：S弯内一共有多少个黄线    
                elif yellow_line_count >= mode3_yellow_final_count:
                        twist.linear.x = 0.0
                        twist.angular.z = 0.0
                        self.cmd_pub.publish(twist)
                        return
                else:
                    yellow_processing = False
 
            # if not self.num_processing and (left_num > 0 and right_num > 0):
            #         print("1111")
            #         # 第一次进入统计，记录开始帧
            #         if self.num_start_frame <= 0:  
            #             self.num_start_frame = frame_count
            #             print(f"开始数字统计，起始帧: {self.num_start_frame}")
            #         elapsed_frames = frame_count - self.num_start_frame
            #         if not self.num_processing:
            #             if elapsed_frames <=3:
            #                 # 左数字统计
            #                 self.stat_left[left_num] = self.stat_left.get(left_num, 0) + 1
            #                 # 右数字统计
            #                 self.stat_right[right_num] = self.stat_right.get(right_num, 0) + 1
            #                 print(f"统计中: frame={elapsed_frames}, left_num={left_num}, right_num={right_num}")

            #             else:
            #                 self.num_processing = True

            #                 # 选频率最高的左数字
            #                 self.left_final_num = max(self.stat_left, key=self.stat_left.get)
            #                 # 选频率最高的右数字
            #                 self.right_final_num = max(self.stat_right, key=self.stat_right.get)
            #                 print(f"统计完成 → 最终左数字: {self.left_final_num}, 最终右数字: {self.right_final_num}")

            #                 # 清空统计数据以便下一次使用
            #                 self.stat_left.clear()
            #                 self.stat_right.clear()
            #                 self.NumSelector.set_target_lane(self.left_final_num, self.right_final_num)
            #                 self.num_start_frame =0
            #         current_output = self.pd_mode1.calc_servo_pd()
            #         twist.linear.x = speed_x*0.5
            #         twist.angular.z = current_output
            #         self.cmd_pub.publish(twist)
            #         return

            # if self.num_processing:
                if not self.NumSelector.update():
                    self.NumSelector.control_motion()
                else:
                    self.num_processing = False
                    left_num = 0
                    right_num = 0
                    left_num_count = 0
                    right_num_count = 0
                    self.left_final_num = None
                    self.right_final_num = None
                    print("数字处理完成")
                return  # 数字处理期间由NumSelector发布命令

            # 正常巡线（既无黄线处理也无数字处理）
            if not self.yellow_processing :   #如果没有黄线也没有数字，就正常巡线。
                print("寻仙")
                current_output = self.pd_mode1.calc_servo_pd()
                twist.linear.x = speed_x 
                twist.angular.z = current_output

        self.cmd_pub.publish(twist)


    def timer_callback(self):
        global yellow_detected_count, yellow_flag
        if yellow_area > yellow_area_limit and yellow_flag == 1:
            print(f"检测到黄线，将要停止： {yellow_area}")
            yellow_detected_count += 1
            yellow_flag = 0
        if yellow_area < 50:
            yellow_flag = 1
        # if yellow_area > 1000:
            #print(f"yellow_area: {yellow_area}")
        self.publish_control_command()                         

    def image_callback(self, msg: Image):
        global frame_count, left_lane, right_lane, center_line, vis_yellow, yellow_area, yellow_detected_count, yellow_flag, num_detected, num_area
        
        frame_count += 1

        # 帧率计算
        # current_time = time.time()
        # if not hasattr(self, 'fps_last_time'):
        #     self.fps_last_time = current_time
        #     self.fps_frame_count = 0
        #     self.current_fps = 0
        
        # self.fps_frame_count += 1
        
        # # 每秒计算一次帧率
        # if current_time - self.fps_last_time >= 1.0:
        #     self.current_fps = self.fps_frame_count / (current_time - self.fps_last_time)
        #     self.get_logger().info(f'VisionNode FPS: {self.current_fps:.2f}')
        #     self.fps_frame_count = 0
        #     self.fps_last_time = current_time

        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.last_frame = frame
        
        if line_flag:
            # 裁减检测车道线部分
            line_img = frame[75:140, :, :]
            # 处理图像
            binary = ImageProcessor.image_process(line_img)
            vis_yellow, yellow_area = ImageProcessor.yellow_detect(frame)
            vis, left_lane, right_lane, center_line = LineProcessor.line_detect(binary)
            #print(f"yellow_detected_count: {yellow_detected_count}")
            cv2.imshow("Lane Lines", vis)
            # cv2.imshow("bianry", binary)
            #cv2.imshow("yellow", vis_yellow)
        # cv2.imshow("VisionNode View", frame)
        cv2.waitKey(1)
    def destroy_node(self):
        self.get_logger().info("Shutting down VisionNode...")
        # self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
