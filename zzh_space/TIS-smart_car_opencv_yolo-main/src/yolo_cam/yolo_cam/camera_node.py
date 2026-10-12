#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import time

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        self.publisher_ = self.create_publisher(Image, 'camera/image_raw', 10)
        self.bridge = CvBridge()

        # --- 摄像头初始化 ---
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

        # 设置摄像头参数：分辨率 + 编码格式 + 目标帧率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # 使用 MJPG 压缩格式
        self.cap.set(cv2.CAP_PROP_FPS, 30)  # 请求30fps
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 减少缓冲，降低延迟

        if not self.cap.isOpened():
            self.get_logger().error('❌ 无法打开摄像头')
            return

        # 读取实际帧率
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.get_logger().info(f'📷 摄像头启动成功: {int(width)}x{int(height)} @ {actual_fps:.2f} FPS')

        # 动态定时器匹配摄像头实际帧率
        interval = 1.0 / (actual_fps if actual_fps > 0 else 30.0)
        self.timer = self.create_timer(interval, self.timer_callback)

        # 帧率统计
        self.frame_count = 0
        self.start_time = time.time()

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warning('⚠️ 无法读取帧')
            return
        frame = cv2.resize(frame, (240, 180), interpolation=cv2.INTER_LINEAR)
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.publisher_.publish(msg)

        # 帧率统计输出
        self.frame_count += 1
        if self.frame_count >= 60:
            elapsed = time.time() - self.start_time
            fps = self.frame_count / elapsed
            self.get_logger().info(f'camera发布帧率: {fps:.2f} FPS')
            self.frame_count = 0
            self.start_time = time.time()

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
