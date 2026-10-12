#!/usr/bin/env python3
import rclpy
import cv2
import numpy as np
import time
import ctypes
import math
import time
from rclpy.node import Node
from sensor_msgs.msg import Imu, Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from yolo_msg.msg import Detection,Detections
from cv_bridge import CvBridge

#检测到障碍物的强拐
SPECIAL_ACTIONS = {
    # # 案例1：模式1下，检测到第2个障碍物时 -> 直行0.5m -> 向右强拐(-0.9) 0.3m
    # 3: {
    #     'trigger_count': 1,
    #     'straight_dist': 0.0,
    #     'turn_force': -0.6,#注意左右
    #     'turn_dist': 0.15
    # },
    
    # # 案例2：模式4下，检测到第3个障碍物时 -> 直行0.4m -> 向左强拐(0.9) 0.3m
    # 5: {
    #     'trigger_count': 1,
    #     'straight_dist': 0.0,
    #     'turn_force': -0.45,
    #     'turn_dist': 0.25
    # }
    # # 其他模式如果没有配置，默认执行普通避障(循线绕行)
}

#转弯变量
turn1 = 0
turn2 = 0# turn 1左；0右
premade =1
#第三阶段有黄线为黄线个数，没黄线为-1
mode3_yellow_count = 1
mode5_yellow_count = 1
#第三阶段黄线个数(包括圆环起点,没有为1，有为1+n
mode3_yellow_final_count = 2
mode5_yellow_final_count = 2

line_thickness = 2    # 绘制车道线宽度
edge_search_width = 130  # 起点搜索仅限距离边缘 20 像素范围内


# 图像处理+八领域变量
road_width = 122      # 道路宽度 + 处理锥桶   
road_width_circle = 121   #有锥桶121 WU 110,,,115,120,127
road_width_num = 117 # 道路宽度 + 处理数字   

#道路偏差
bias_circle = 9
bias_num = 1 #9
line_mode = 1


# ============================= 数字处理 =============================
select_distance_turn1 = 0.08  #数字选择强拐
select_distance_turn2 = 0.03   #数字选择直行
select_distance_turn3 = 0.03
select_distance = 0.18       # 巡线里程计X2

# distance_turn1 = 0.072 #数字选择强拐
# distance_turn2 = 0.09

select_turn = 0.4 #强拐角度z
num_speed = 0.9 #识别到数字速度比例系数

# ============================= Red/Cone 避障参数 =============================
red_avoid_distance = 0.7    # 红线里程计
cone_avoid_distance = 0.3  # 锥桶里程计
red_speed_ratio = 1.2       # 避障时的速度系数

#circle
circle_odom_c1= 1.36   #1.35
yellow_straight_c1 = 0.55 #起步直行距离

#！！若turn2=1(第二个圈左拐)，则需要改成注释里的值（1.13/0.35/0.28/0.13由上往下按顺序）
circle_odom_c2 = 1.35 #1.13左  1.35右
yellow_straight_c2 = 0.27 #0.35左  0.17右
circle_out_trun_c1 = 0.2   #0.28左  0.27右
circle_out_trun_c2 = 0.24   #0.13左   0.27右
#！！！turn2 = 1！！！！！
circle_in_trun = 0.33

#servo_pd
error_p = -0.009
Kp = 1.1
Kd = 0.4 #0.4
Kp_c = 1.0
Kd_c = 0.4
pid_add = 1.1
error_limit = 0.75

#area
red_area_limit = 2000.0
cone_area_limit = 4000.0
yellow_area_limit = 2100 #1800
num_area_limit = 601 ##1300

# 控制参数0.2
speed_x = 0.25  # 基础前进速度

# speed_x_red = 0.15


# =============================
# 全局颜色范围参数(HSV)
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

#红色
left_red = 0
right_red = 0
left_cone = 1
right_cone = 0

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


clear_flag = 0
# 数字检测增强
left_num_count = 0  # 左侧数字检测计数
right_num_count = 0  # 右侧数字检测计数
num_decision_made = False  # 是否已做出数字决策
num_processing = False

# 红色障碍物标志
right_flag = 0
left_flag = 0

frame_count = 0  # 总帧数计数器
current_x = 0
total_distance = 0.0


class ImageProcessor:
    @staticmethod
    def image_process(hsv):
        # if yolo_num_boxes is not None:
        #     for (x1, y1, x2, y2) in yolo_num_boxes:
        #         # 映射 YOLO 坐标到 line_img 局部坐标
        #         local_y1 = max(0, int(y1 - 75))
        #         local_y2 = min(origin_img.shape[0], int(y2 - 75))
        #         local_x1 = max(0, int(x1))
        #         local_x2 = min(origin_img.shape[1], int(x2))

        #         # 检查是否与 [75,140) 区域重叠
        #         if local_y2 > local_y1:
        #             # 将检测区域设为白色
        #             binary[local_y1:local_y2, local_x1:local_x2] = 255

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

        # if yolo_num_boxes is not None:
        #     for (x1, y1, x2, y2) in yolo_num_boxes:
        #         # 映射 YOLO 坐标到 line_img 局部坐标
        #         local_y1 = max(0, int(y1 - 75))
        #         local_y2 = min(origin_img.shape[0], int(y2 - 75))
        #         local_x1 = max(0, int(x1))
        #         local_x2 = min(origin_img.shape[1], int(x2))

        #         # 检查是否与 [75,140) 区域重叠
        #         if local_y2 > local_y1:
        #             # 将检测区域设为白色
        #             binary[local_y1:local_y2, local_x1:local_x2] = 255

        # 去噪(开闭操作)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # 检查白色区域占比(如果太小则返回空图)
        # white_pixels = np.sum(binary > 0)
        # if white_pixels < 100:  # 可以根据图像大小调整
        #     return np.zeros_like(binary)

        return binary


    @staticmethod
    def yellow_detect(hsv):

        # 黄色区域掩码
        mask_yellow = cv2.inRange(hsv, LOWER_YELLOW, UPPER_YELLOW)

        # 计算黄色像素个数
        yellow_pixels = int(np.sum(mask_yellow > 0))

        # 可视化：画轮廓
        contours, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(hsv, contours, -1, (0, 255, 255), 2)

        return hsv, yellow_pixels

class LineProcessor:
    @staticmethod
    def find_lane_start(binary_image, side="left"):
        h, w = binary_image.shape
        margin_width = 5

        if side == "left":
            x_start, x_end = margin_width,edge_search_width
        else:
            x_start, x_end = w - edge_search_width,w - margin_width

        #  边缘三列检测 (预检查)
        if side == "left":
            # 先检查是否全白
            black_pixel_count = 0

            # 检查第一列 (x = 0)
            for y in range(h-1):
                if binary_image[y, 0] == 0:  # 如果是黑色像素
                    black_pixel_count += 1
                if black_pixel_count > 1:  # 如果黑色像素超过 1 个，直接跳出
                    break
            if black_pixel_count <= 1:  # 如果黑色像素总数 <= 1
                return None  # 如果全是白，直接认为该侧无效，返回空

            for y in range(h-1, -1, -1):
                row = binary_image[y, :]
                for x in range(x_start-1):
                    if int(row[x]) == 255 :
                        break
                    if int(row[x]) == 0 and row[x+1] == 255:
                        return (y, x+1)

        elif side == "right":
            # 先检查是否全白
            black_pixel_count = 0

            # 检查第一列 (x = 0)
            for y in range(h-1):
                if binary_image[y, w-1] == 0:  # 如果是黑色像素
                    black_pixel_count += 1
                if black_pixel_count > 1:  # 如果黑色像素超过 1 个，直接跳出
                    break
            if black_pixel_count <= 1:  # 如果黑色像素总数 <= 1
                return None  # 如果全是白，直接认为该侧无效，返回空
            
            for y in range(h-1, -1, -1):
                row = binary_image[y, :]
                for x in range(w-1, x_end-1, -1):
                    if int(row[x]) == 255 :
                        break
                    if int(row[x]) == 0 and row[x-1] == 255:
                        return (y, x-1)


        for y in range(h-1, -1, -1):
            row = binary_image[y, :]
            if side == "left":
                for x in range(x_start-1, x_end-1):
                    if int(row[x]) == 255 :
                        break
                    if int(row[x]) == 0 and row[x+1] == 255:
                        return (y, x+1)
            else:
                for x in range(x_end-1, x_start-1, -1):
                    if int(row[x]) == 255 :
                        break
                    if int(row[x]) == 0 and row[x-1] == 255:
                        return (y, x-1)
        return None


        h, w = binary_image.shape
        if start_point is None:
            return []

        visited = np.zeros_like(binary_image, dtype=bool)
        #[(-1, 0), (-1, 1), (0, 1), (0, -1), (-1, -1)]
        directions = [ (-1, 1), (0, -1), (-1, -1)]
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


        #寻一条线逻辑
        if current_mode == 2 or current_mode == 4:
            if cone_detected:
                width = road_width
            else:
                width = road_width_circle
        
        if line_mode == 3 and clear_flag:
            left_pts.clear()
            print(f"靠右行驶")
        if line_mode == 2 and clear_flag:
            right_pts.clear()
            print(f"靠左行驶")

        
        if num_processing:
            width = road_width_num
        else:
            width = road_width

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
    
    @staticmethod
    def trace_line(binary_image, start_point, side="left"):
        h, w = binary_image.shape
        if start_point is None:
            return []

        visited = np.zeros_like(binary_image, dtype=bool)
        directions = [(-1, 0), (-1, 1), (0, 1), (0, -1), (-1, -1)]
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
        global error, road_width, left_lane, right_lane, center_line
        global line_mode

        
        line_mode = self.mode

        if not center_line:
            return 0.0

        top_x = center_line[-1][0]
        mid_x = center_line[len(center_line) // 2][0]
        bottom_x = center_line[0][0]

        if current_mode == 2 or current_mode == 4:
            bias = bias_circle
        else:
            bias = bias_num
        
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
        # if error < error_limit:
        #     Kp = self.Kp
        #     Kd = self.Kd
        # else:
        #     Kp = self.Kp * pid_add
        #     Kd = self.Kd * pid_add

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
        self.target_lane = None  # 目标车道('left' or 'right')
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
                print(f"左走: left_final_num={left_final_num} >= right_final_num={right_final_num}")
            else:
                self.target_lane = 'right'
                print (f"error:{error}")
                print(f"右走: right_final_num={right_final_num} > left_final_num={left_final_num}")
        elif left_final_num !=0 :
            if left_final_num >=5 :
                self.target_lane = 'left'
                print (f"error:{error}")
                print(f"(没看到右边数字)左走: left_final_num={left_final_num} ")
            else:
                self.target_lane = 'right'
                print (f"error:{error}")
                print(f"(没看到右边数字)右走: right_final_num={right_final_num} ")
        else :
            if right_final_num >=5 :
                self.target_lane = 'right'
                print (f"error:{error}")
                print(f"(没看到左边数字)右走: right_final_num={right_final_num} ")
            else:
                self.target_lane = 'left'
                print (f"error:{error}")
                print(f"(没看到左边数字)左走: left_final_num={left_final_num} ")

        #11111111111111111111111111111111111111111111111111111
        if current_mode ==3:
            self.target_lane = 'left'

        if current_mode ==5:
            self.target_lane = 'right'
        # 初始化阶段
        self.stage = 0  # 从第一阶段开始
        self.start_x = total_distance  # 等待 update() 记录开始位置
        self.distance_travelled = 0

    def update(self):
        global total_distanc,clear_flag

        if self.target_lane is None:
            return True   # 没有目标，直接退出

        # 当前行驶距离
        self.distance_travelled = abs(total_distance - self.start_x)
        
        # 阶段切换逻辑
        if self.stage == 0 and self.distance_travelled > select_distance_turn1:
            self.stage = 1  # 切换到巡线轻拐阶段
            print(f"进入{self.target_lane}222巡线轻拐阶段")
            self.start_x = total_distance  # 重置起始距离
            self.distance_travelled = 0
            
        elif self.stage == 1 and self.distance_travelled > select_distance_turn2:
            self.stage = 2  # 切换到反向归正阶段
            print(f"进入{self.target_lane}333归正阶段")
            self.start_x = total_distance  # 重置起始距离
            self.distance_travelled = 0
            
        elif self.stage == 2 and self.distance_travelled > select_distance_turn3:
            clear_flag = 1
            self.stage = 3  # 切换到巡线归正阶段
            print(f"进入{self.target_lane}444巡线归正阶段")
            self.start_x = total_distance  # 重置起始距离
            self.distance_travelled = 0
            
        elif self.stage == 3 and self.distance_travelled > select_distance:
            print("end数字切道完成，退出模式")
            self.start_x = 0
            self.target_lane = None
            self.stage = 0
            clear_flag = 0
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
                # print("左走巡左线轻拐阶段")
                twist.linear.x = speed_x * num_speed + 0.03
                twist.angular.z = select_turn/2
            else:  # right
                # print("右走巡右线轻拐阶段")
                twist.linear.x = speed_x * num_speed + 0.03
                twist.angular.z = -select_turn/2

        # 阶段2：反向归正阶段
        elif self.stage == 2:
            if self.target_lane == 'left':
                # print("左走右转归正阶段")
                twist.linear.x = speed_x * num_speed
                twist.angular.z = select_turn/3
            else:  # right
                # print("右走左转归正阶段")
                twist.linear.x = speed_x * num_speed
                twist.angular.z = -select_turn/3

        # 阶段3：巡线归正阶段
        elif self.stage == 3:
            if self.target_lane == 'left':
                # print("左走巡左线归正阶段")
                current_output = self.pd_mode2.calc_servo_pd()  # 循左线
                twist.linear.x = speed_x * num_speed + 0.03
                twist.angular.z = current_output
            else:  # right
                # print("右走巡右线归正阶段")
                current_output = self.pd_mode3.calc_servo_pd()  # 循右线
                twist.linear.x = speed_x * num_speed + 0.03
                twist.angular.z = current_output

        self.cmd_pub.publish(twist)

class Red:
    def __init__(self):
        self.active = False          # 是否处于动作执行状态
        self.start_x = 0.0           # 动作开始时的里程计位置
        self.target_lane = None      # 普通避障：'follow_left' 或 'follow_right'
        
        # --- 修改点1：改为按模式计数的字典 ---
        # 格式: {模式1: 计数, 模式4: 计数, ...}
        self.mode_counts = {} 
        
        # 特殊动作状态机
        self.is_special_action = False # 当前是否在做特殊动作
        self.special_phase = 0         # 0:直行/循线阶段, 1:强拐阶段
        
        # 当前正在执行的特殊动作参数缓存
        self.current_special_params = {}

        # 初始化独立的PD控制器
        self.pd_left = Servo_PD(servo_p=Kp, servo_d=Kd)
        self.pd_left.set_mode(2)     # 循左线
        self.pd_right = Servo_PD(servo_p=Kp, servo_d=Kd)
        self.pd_right.set_mode(3)    # 循右线
        self.pd_normal = Servo_PD(servo_p=Kp, servo_d=Kd)
        self.pd_normal.set_mode(1)   # 循中线(辅助直行)


    def update(self, current_mode_in):
        """
        返回 Twist: 如果需要执行动作，返回控制指令
        返回 None:  如果无任务，返回 None
        """
        global total_distance, speed_x
        global red_detected, cone_detected
        global left_red, right_red, left_cone, right_cone
        global SPECIAL_ACTIONS # 引用全局配置
        
        # =========================================
        # 1. 触发逻辑
        # =========================================
        if not self.active:
            if red_detected or cone_detected:
                
                # --- 初始化该模式的计数器 ---
                if current_mode_in not in self.mode_counts:
                    self.mode_counts[current_mode_in] = 0
                
                # --- 计数加 1 ---
                self.mode_counts[current_mode_in] += 1
                current_count = self.mode_counts[current_mode_in]
                
                print(f"!!! 模式 {current_mode_in} 障碍物计数: {current_count}")
                
                self.active = True
                self.start_x = total_distance
                
                # --- 检查是否匹配特殊动作配置 ---
                # 判断条件：当前模式在配置表中 且 计数 == 触发计数
                if (current_mode_in in SPECIAL_ACTIONS and 
                    current_count == SPECIAL_ACTIONS[current_mode_in]['trigger_count']):
                    
                    # >>> 触发特殊动作 <<<
                    self.is_special_action = True
                    self.special_phase = 0
                    # 加载配置参数
                    self.current_special_params = SPECIAL_ACTIONS[current_mode_in]
                    
                    print(f"--> 触发模式 {current_mode_in} 特殊动作！")
                    print(f"--> 计划: 直行 {self.current_special_params['straight_dist']}m -> 强拐 {self.current_special_params['turn_force']}")
                    
                    return self._get_special_motion()
                
                else:
                    # >>> 普通避障模式 (计数不匹配 或 该模式无配置) <<<
                    self.is_special_action = False
                    
                    # 优先级判断：右侧障碍 -> 循左
                    if (red_detected and right_red) or (cone_detected and right_cone):
                        self.target_lane = 'follow_left'
                        print(f"--> 普通避障(Mode {current_mode_in}, #{current_count}): 右侧障碍，循左线")
                    # 左侧障碍 -> 循右
                    elif (red_detected and left_red) or (cone_detected and left_cone):
                        self.target_lane = 'follow_right'
                        print(f"--> 普通避障(Mode {current_mode_in}, #{current_count}): 左侧障碍，循右线")
                    else:
                        self.target_lane = 'follow_left'
                    
                    return self._get_normal_motion()
            
            return None

        # =========================================
        # 2. 执行逻辑
        # =========================================
        else:
            if self.is_special_action:
                return self._update_special_action()
            else:
                return self._update_normal_avoidance()

    def _update_normal_avoidance(self):
        global total_distance, red_detected, left_red, right_red
        distance_traveled = abs(total_distance - self.start_x)

        if red_detected or left_red or right_red:
            avoid_dist = red_avoid_distance
        else:
            avoid_dist = cone_avoid_distance           

        if distance_traveled < avoid_dist:
            return self._get_normal_motion()
        else:
            print(f"普通避障完成")
            self.active = False
            return None

    def _get_normal_motion(self):
        twist = Twist()
        twist.linear.x = speed_x * red_speed_ratio
        if self.target_lane == 'follow_left':
            twist.angular.z = self.pd_left.calc_servo_pd()
        elif self.target_lane == 'follow_right':
            twist.angular.z = self.pd_right.calc_servo_pd()
        return twist

    def _update_special_action(self):
        """执行特殊动作(使用 self.current_special_params 中的参数)"""
        global total_distance, speed_x
        
        dist_now = abs(total_distance - self.start_x)
        twist = Twist()
        
        # 获取当前配置的参数
        straight_limit = self.current_special_params['straight_dist']
        turn_force = self.current_special_params['turn_force']
        turn_dist_limit = self.current_special_params['turn_dist']
        
        # --- 阶段 0: 直行 ---
        if self.special_phase == 0:
            if dist_now < straight_limit:
                twist.linear.x = speed_x
                # 使用循中线保持直行稳定性，或者直接给 0.0
                twist.angular.z = self.pd_normal.calc_servo_pd() 
                return twist
            else:
                self.special_phase = 1
                self.start_x = total_distance # 重置距离
                print(f"特殊动作: 直行结束，开始强拐")
                return self._update_special_action()
                
        # --- 阶段 1: 强拐 ---
        elif self.special_phase == 1:
            dist_turn = abs(total_distance - self.start_x)
            
            if dist_turn < turn_dist_limit:
                twist.linear.x = speed_x * 0.8
                twist.angular.z = turn_force # 使用配置的力度
                # print(f"特殊强拐: {dist_turn:.2f}/{turn_dist_limit}")
                return twist
            else:
                print(f"特殊动作全部完成")
                self.active = False
                self.is_special_action = False
                self.current_special_params = {} # 清空参数
                return None

    def _get_special_motion(self):
        return self._update_special_action()

class CircleController:
    def __init__(self, cmd_pub):
        self.stage = 1
        self.start_frame = frame_count
        self.cmd_pub = cmd_pub

        # 初始化PD控制器
        self.pd_mode1 = Servo_PD(servo_p=Kp_c, servo_d=Kd_c)#循中线
        self.pd_mode1.set_mode(1)
        self.pd_mode2 = Servo_PD(servo_p=Kp_c, servo_d=Kd_c)#循左线
        self.pd_mode2.set_mode(2)
        self.pd_mode3 = Servo_PD(servo_p=Kp_c, servo_d=Kd_c)#循右线
        self.pd_mode3.set_mode(3)

    def update(self):
        """更新当前阶段并返回是否完成"""
        global frame_count, current_x, total_distance, speed_x
        global left_num,right_num, right_flag, left_flag, circle_odom
        twist = Twist()
        if self.stage == 1:

            # elapsed_frames = frame_count - self.start_frame
            # if elapsed_frames < circle_in_trun:

            distance_traveled_1 = abs(total_distance  - current_x)
            if distance_traveled_1 < circle_in_trun:
                if right_flag:
                    twist.linear.x = speed_x
                    twist.angular.z = -0.9
                elif left_flag:
                    twist.linear.x = speed_x
                    twist.angular.z = 0.9

            else:
                # 进入第二阶段
                self.stage = 2
                # self.start_frame = frame_count
                current_x = total_distance
                print("二阶段：循线")

        elif self.stage == 2:
            # 第二阶段：循线行驶
            if left_flag:
                current_output = self.pd_mode1.calc_servo_pd()  # 循又线
                twist.linear.x = speed_x
                twist.angular.z = current_output
                if abs(total_distance  - current_x) == 0.1:
                    print(f"二阶段循右线")
            elif right_flag:
                current_output = self.pd_mode1.calc_servo_pd()  # 循作
                twist.linear.x = speed_x
                twist.angular.z = current_output
                if abs(total_distance  - current_x) == 0.1:
                    print(f"二阶段循左线")

            # 检查是否到达切换条件
            distance_traveled_2 = abs(total_distance  - current_x)
            if current_mode == 4:
                circle_odom =circle_odom_c2
                print("111d")
            else:
                circle_odom = circle_odom_c1
                print("222d")
            if distance_traveled_2 > circle_odom:  # 行驶了1米后进入第三阶段
                self.stage = 3
                # self.start_frame = frame_count
                current_x = total_distance
                if abs(total_distance  - current_x) == 0.1:
                    print("三阶段：出圈强转")

        elif self.stage == 3:
            # 第三阶段：出圈强转
            # elapsed_frames = frame_count - self.start_frame
            # if elapsed_frames < circle_out_trun:

            distance_traveled_3 = abs(total_distance  - current_x)
            if current_mode ==4:
                circle_out_trun =circle_out_trun_c2
            else:
                circle_out_trun =circle_out_trun_c1
            if distance_traveled_3 < circle_out_trun:
                if right_flag:
                    twist.linear.x = speed_x
                    twist.angular.z = -0.75
                    print(f"三阶段强转右")
                elif left_flag:
                    twist.linear.x = speed_x
                    twist.angular.z = 0.75
                    print(f"三阶段强转左")
            else:
                # 完成整个圆圈
                print("圆圈完成")
                right_flag = 0
                left_flag = 0
                return True  # 返回完成标志

        self.cmd_pub.publish(twist)
        return False  # 未完成

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
        self.Red_processor = Red()  # 
        # 创建定时器(30Hz)
        self.timer = self.create_timer(1/30.0, self.timer_callback)
        # 黄线处理状态
        self.yellow_line_count_mode3 = 0
        self.yellow_line_count_mode5 = 0
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

        # 圆圈控制器
        self.circle_controller = None

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
        global left_num, right_num, right_cone, left_cone, right_red, left_red
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
                    # if ls_name == "1" or ls_name == "s1":
                    #     value = 1
                    # elif ls_name == "2" or ls_name == "s2":
                    #     value = 2
                    # elif ls_name == "3" or ls_name == "s3":
                    #     value = 3
                    # elif ls_name == "4" or ls_name == "s4":
                    #     value = 4
                    # elif ls_name == "5" or ls_name == "s5":
                    #     value = 5
                    # elif ls_name == "6" or ls_name == "s6":
                    #     value = 6
                    # elif ls_name == "7" or ls_name == "s7":
                    #     value = 7
                    # elif ls_name == "8" or ls_name == "s8":
                    #     value = 8

                    if ls_name == "s1":
                        value = 1
                    elif ls_name == "s2":
                        value = 2
                    elif ls_name == "s3":
                        value = 3
                    elif ls_name == "s4":
                        value = 4
                    elif ls_name == "s5":
                        value = 5
                    elif ls_name == "s6":
                        value = 6
                    elif ls_name == "s7":
                        value = 7
                    elif ls_name == "s8":
                        value = 8

                    # 根据位置判断左右
                    if (x1+x2)/2 <= 110:
                        temp_left_num = value
                        # print(f"left:{value}:{(x1+x2)/2}")
                    if (x1+x2)/2 > 130:
                        temp_right_num = value
                        # print(f"rightt:{value}:{(x1+x2)/2}")

            if ls_name == "danger_red":
                red_area = abs((x2-x1)*(y2-y1))
                if red_area > red_area_limit:
                    # print(f"Class: {det.cls}, Confidence: {det.conf:.2f}, red_area: {red_area}, 位置:{(x1+x2)/2}")
                    local_red_detected = 1
                    if (x1+x2)/2 <= 120:
                        left_red = 1
                        right_red = 0
                    if (x1+x2)/2 > 120:
                        right_red = 1
                        left_red = 0

            if ls_name == "cone":
                cone_area = abs((x2-x1)*(y2-y1))
                if cone_area > cone_area_limit:
                    #print(f"Class: {det.cls}, Confidence: {det.conf:.2f}, cone_area: {cone_area}, 位置:{(x1+x2)/2}")
                    local_cone_detected = 1
                    if (x1+x2)/2 <= 120:
                        left_cone = 1
                        right_cone = 0
                    if (x1+x2)/2 > 120:
                        right_cone = 1
                        left_cone = 0

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
            # 没有检测到数字，递减计数(逐渐失效)
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
            # print(f"距离：{self.total_distance}")
            global total_distance
            total_distance = self.total_distance

        self.last_odom_x = x
        self.last_odom_y = y

    def publish_control_command(self):
        global current_mode, current_x, yellow_detected_count, speed_x, frame_count, num_detected
        global left_num, right_num, left_num_count, right_num_count, num_decision_made
        global cone_detected, red_detected, left_flag, right_flag
        global num_processing
        num_processing = self.num_processing
        twist = Twist()

        if current_mode != 2 and current_mode != 4 and not self.num_processing:
            avoidance_twist = self.Red_processor.update(current_mode)
            if avoidance_twist is not None:
                self.cmd_pub.publish(avoidance_twist)
                return 

        if current_mode == 1:
            # 检测到黄线且尚未开始处理
            if yellow_detected_count > 0 and not self.yellow_processing:
                self.yellow_processing = True
                self.yellow_start_frame = frame_count
                yellow_detected_count = 0

            # 黄线处理中
            if self.yellow_processing:
                elapsed_frames = frame_count - self.yellow_start_frame

                if elapsed_frames < 12:
                    # 停车期间识别数字
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0

                    # 在停车后期(第10帧后)根据数字做决策
                    if elapsed_frames >= 4 and not num_decision_made:
                        if premade == 0:
                            if left_num > 0 and right_num > 0:
                                # 根据数字大小选择方向(数字大的走，相等走左边)
                                if left_num >= right_num:
                                    left_flag = 1
                                    right_flag = 0
                                    print(f"黄线停车识别-第一圈走左: left_num={left_num} >= right_num={right_num}")
                                else:
                                    right_flag = 1
                                    left_flag = 0
                                    print(f"黄线停车识别-第一圈走右: right_num={right_num} > left_num={left_num}")
                                num_decision_made = True
                            else:
                                # 如果到第12帧还没识别到数字，使用默认策略
                                if elapsed_frames >= 10:
                                    left_flag = 1  # 默认左圈
                                    right_flag = 0
                                    num_decision_made = True
                                    print(f"黄线停车未识别到数字-默认左转")
                        else :
                            left_flag = turn1
                            right_flag = int(not turn1) #右边取反 turn 1左；0右
                            print(f"turn ={turn1}，turn1 1左；0右//黄线停车识别: right_num={right_num} : left_num={left_num}")
                            num_decision_made = True

                # elif elapsed_frames < yellow_straight_c1:
                elif total_distance  < yellow_straight_c1:
                    # 启动加速阶段fcz
                    twist.linear.x = speed_x
                    twist.angular.z = 0.0
                else:
                    current_x = total_distance
                    current_mode = 2
                    self.yellow_processing = False
                    # 进入模式2时立即创建圆圈控制器
                    self.circle_controller = CircleController(self.cmd_pub)
                    # 清零数字计数
                    left_num = 0
                    right_num = 0
                    left_num_count = 0
                    right_num_count = 0
                    print(f"进入模式2")

            else:
                # 正常行驶
                #print(f"正常行驶")
                twist.linear.x = speed_x
                twist.angular.z = 0.0

        elif current_mode == 2:
            # 第二阶段：执行圆圈动作
            if self.circle_controller is not None:
                # 更新圆圈控制器，如果返回True表示完成
                if self.circle_controller.update():
                    current_mode = 3
                    print(f"进入模式3")
                    yellow_detected_count = 0  #置零，以便模式3时黄线检测！！
                    self.circle_controller = None
                    num_decision_made = False  # 重置决策标志，为第二个圆圈做准备
                    left_num = 0
                    right_num = 0
                    left_num_count = 0
                    right_num_count = 0
            else:
                # 如果没有控制器，创建一個
                self.circle_controller = CircleController(self.cmd_pub)

        elif current_mode == 3:
            # 先检查黄线(最高优先级)
            #print(f":yellow_detected_count:{yellow_detected_count},yellow_processing:{self.yellow_processing}:")
            if yellow_detected_count > 0 and not self.yellow_processing:
                self.yellow_line_count_mode3 += 1  # 计数
                self.yellow_processing = True
                self.yellow_start_frame = frame_count
                yellow_detected_count = 0
                print(f"检测到黄线，当前计数: {self.yellow_line_count_mode3}")

            # 黄线处理中(最高优先级)
            if self.yellow_processing:
                
                # print(f"处理黄线中，计数{self.yellow_line_count_mode3}，帧数{elapsed_frames},left_num={left_num} : right_num={right_num}")
    
                # 第一根黄线
                if mode3_yellow_count >= self.yellow_line_count_mode3 > 0:
                    elapsed_frames = frame_count - self.yellow_start_frame
                    if elapsed_frames < 17:
                        twist.linear.x = 0.0
                        twist.angular.z = 0.0
                        self.cmd_pub.publish(twist)

                        return
                    else:
                        # 第一根黄线处理完成，继续巡线
                        self.yellow_processing = False
                        print(f"第{self.yellow_line_count_mode3}根黄线处理完成")

                # 黄线(第二个圆圈前，需要识别数字)
                elif self.yellow_line_count_mode3 >= mode3_yellow_final_count :
                    elapsed_frames = frame_count - self.yellow_start_frame
                    if elapsed_frames < 15:
                        # 停车期间识别数字
                        current_x = total_distance
                        twist.linear.x = 0.0
                        twist.angular.z = 0.0
                        self.cmd_pub.publish(twist)
                        print("1")

                        # 在停车后期根据数字做决策
                        if elapsed_frames >= 4 and not num_decision_made:
                            if premade == 0:
                                if left_num > 0 and right_num > 0:
                                    # 根据数字大小选择方向(数字大的走，相等走左边)
                                    if left_num >= right_num:
                                        left_flag = 1
                                        right_flag = 0
                                        print(f"黄线停车识别-第一圈走左: left_num={left_num} >= right_num={right_num}")
                                    else:
                                        right_flag = 1
                                        left_flag = 0
                                        print(f"黄线停车识别-第一圈走右: right_num={right_num} > left_num={left_num}")
                                    num_decision_made = True
                                else:
                                    # 如果到第12帧还没识别到数字，使用默认策略
                                    if elapsed_frames >= 10:
                                        left_flag = 1  # 默认左圈
                                        right_flag = 0
                                        num_decision_made = True
                                        print(f"黄线停车未识别到数字-默认左转")
                            else:
                                left_flag = turn2
                                right_flag = int(not turn2)  # 右边取反 turn 1左；0右
                                print(
                                    f"turn ={turn2}，turn2 1左；0右//黄线停车识别: right_num={right_num} : left_num={left_num}")
                                num_decision_made = True
                        return

                    elif abs(total_distance - current_x) < yellow_straight_c2:
                        # 启动加速阶段
                        twist.linear.x = speed_x
                        twist.angular.z = 0.0
                        self.cmd_pub.publish(twist)
                        return
                    else:
                        # 第二根黄线处理完成，进入模式4
                        current_x = total_distance
                        current_mode = 4
                        self.yellow_processing = False
                        self.circle_controller = CircleController(self.cmd_pub)
                        # 清零数字计数
                        left_num = 0
                        right_num = 0
                        left_num_count = 0
                        right_num_count = 0
                        print("进入模式4：第二圈")
                        return
                else:
                    current_output = self.pd_mode3.calc_servo_pd()
                    twist.linear.x = 0.15
                    twist.angular.z = current_output
                    self.cmd_pub.publish(twist)
                    return
            # 数字处理
            if not self.num_processing and (left_num > 0 or right_num > 0):
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
                twist.linear.x = 0.0
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

            # 正常巡线(既无黄线处理也无数字处理)
            if not self.yellow_processing and not self.num_processing:
                current_output = self.pd_mode3.calc_servo_pd()
                twist.linear.x = speed_x
                twist.angular.z = current_output

        elif current_mode == 4:
            # 第四阶段：执行第二个圆圈动作
            if self.circle_controller is not None:
                # 更新圆圈控制器，如果返回True表示完成
                if self.circle_controller.update():
                    current_mode = 5
                    yellow_detected_count = 0  #置零，以便模式5时黄线检测停止，完成巡线！！
                    self.circle_controller = None
                    num_decision_made = False  # 重置决策标志
                    left_num = 0
                    right_num = 0
                    left_num_count = 0
                    right_num_count = 0
            else:
                # 如果没有控制器，创建一個
                self.circle_controller = CircleController(self.cmd_pub)

        elif current_mode == 5:
            if yellow_detected_count > 0 and not self.yellow_processing:
                self.yellow_line_count_mode5 += 1  # 计数
                self.yellow_processing = True
                self.yellow_start_frame = frame_count
                yellow_detected_count = 0
                print(f"检测到黄线，当前计数: {self.yellow_line_count_mode5}")

            # 黄线处理中(最高优先级)
            if self.yellow_processing:
                elapsed_frames = frame_count - self.yellow_start_frame
                # print(f"处理黄线中，计数{self.yellow_line_count_mode5}，帧数{elapsed_frames},left_num={left_num} : right_num={right_num}")

                # 第一根黄线
                if mode5_yellow_count >= self.yellow_line_count_mode5 > 0:
                    if elapsed_frames < 17:
                        twist.linear.x = 0.0
                        twist.angular.z = 0.0
                        self.cmd_pub.publish(twist)
                        return
                    else:
                        # 第一根黄线处理完成，继续巡线
                        self.yellow_processing = False
                        print(f"第{self.yellow_line_count_mode5}根黄线处理完成")

                # 黄线(第二个圆圈前，需要识别数字)
                elif self.yellow_line_count_mode5 >= mode5_yellow_final_count:
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0
                    self.cmd_pub.publish(twist)
                    return

            # 数字处理
            if not self.num_processing and (left_num > 0 or right_num > 0):
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
                twist.linear.x = 0.0
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

            # 黄线处理中
            if not self.yellow_processing:
                # 正常巡线行驶
                current_output = self.pd_mode1.calc_servo_pd()
                twist.linear.x = speed_x
                twist.angular.z = current_output

        # 发布命令
        if current_mode != 2 and current_mode != 4 and self.num_processing == 0 :  # 模式2，4由CircleController自己发布命令。模式3,5检测到数字时候也由NumSelector自己发布命令
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
        # print(f"yellow_area: {yellow_area}")
        self.publish_control_command()

    def image_callback(self, msg: Image):
        global frame_count, left_lane, right_lane, center_line, vis_yellow, yellow_area, yellow_detected_count, yellow_flag, num_detected, num_area

        frame_count += 1

        # # 帧率计算
        # current_time = time.time()
        # if not hasattr(self, 'fps_last_time'):
        #     self.fps_last_time = current_time
        #     self.fps_frame_count = 0
        #     self.current_fps = 0

        # self.fps_frame_count += 1

        # 每秒计算一次帧率
        # if current_time - self.fps_last_time >= 1.0:
        #     self.current_fps = self.fps_frame_count / (current_time - self.fps_last_time)
        #     self.get_logger().info(f'VisionNode FPS: {self.current_fps:.2f}')
        #     self.fps_frame_count = 0
        #     self.fps_last_time = current_time

        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.last_frame = frame

        if line_flag:
            # 裁减检测车道线部分
            hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
            line_img = hsv[75:140, :, :]
            # 处理图像
            binary = ImageProcessor.image_process(line_img)
            vis_yellow, yellow_area = ImageProcessor.yellow_detect(hsv)
            vis, left_lane, right_lane, center_line = LineProcessor.line_detect(binary)
            # print(f"yellow_detected_count: {yellow_detected_count}")
            # cv2.imshow("Lane Lines", vis)
            # cv2.imshow("bianry", binary)
            # cv2.imshow("yellow", vis_yellow)
        # cv2.imshow("VisionNode View", frame)
        # cv2.waitKey(1)
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
