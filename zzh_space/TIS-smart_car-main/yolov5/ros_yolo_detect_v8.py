#!/usr/bin/env python3
# coding=utf-8
import sys
import os
import pathlib
import time
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import torch
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO

TARGET_DIGITS = ['1', '2', '3', '4', '5', '6', '7', '8', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8']

class RosYOLODetector(Node):
    def __init__(self):
        super().__init__('ros_yolo_detector')
        default_model_path = str(pathlib.Path(__file__).resolve().parent / 'best_yolov8.pt')
        self.declare_parameter('model_path', default_model_path)
        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('detection_topic', '/hobot_dnn_detection')
        self.declare_parameter('conf_threshold', 0.15)
        self.declare_parameter('iou_threshold', 0.45)
        self.declare_parameter('skip_frames', 2)
        self.declare_parameter('target_digits', TARGET_DIGITS)
        self.declare_parameter('publish_placeholder', True)

        model_path = self.get_parameter('model_path').value
        image_topic = self.get_parameter('image_topic').value
        detection_topic = self.get_parameter('detection_topic').value
        self.conf_threshold = float(self.get_parameter('conf_threshold').value)
        self.iou_threshold = float(self.get_parameter('iou_threshold').value)
        self.skip_frames = max(0, int(self.get_parameter('skip_frames').value))
        self.publish_placeholder = bool(self.get_parameter('publish_placeholder').value)
        target_digits = self.get_parameter('target_digits').value
        if isinstance(target_digits, str):
            target_digits = [item.strip() for item in target_digits.split(',') if item.strip()]
        self.target_digits = list(target_digits) if target_digits else TARGET_DIGITS
        
        # 检查CUDA可用性
        if torch.cuda.is_available():
            self.get_logger().info('CUDA available, using GPU')
            device = 'cuda'
        else:
            self.get_logger().warn('CUDA not available, using CPU')
            device = 'cpu'
        
        self.get_logger().info(f'Loading YOLOv8 model from {model_path}...')
        self.model = YOLO(model_path)
        self.model.to(device)
        
        self.get_logger().info(f'Model classes: {self.model.names}')
        self.get_logger().info(f'Running on device: {self.model.device}')
        
        self.bridge = CvBridge()
        self.enable_vis = False
        self.frame_count = 0
        
        self.image_sub = self.create_subscription(Image, image_topic, self.image_callback, 10)
        self.detection_pub = self.create_publisher(Detection2DArray, detection_topic, 10)
        
        self.get_logger().info(
            f'YOLOv8 detector started: image_topic={image_topic}, '
            f'detection_topic={detection_topic}, conf={self.conf_threshold}, '
            f'iou={self.iou_threshold}, skip_frames={self.skip_frames}')

    def image_callback(self, ros_img):
        self.frame_count += 1
        if self.frame_count % (self.skip_frames + 1) != 0:
            return
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(ros_img, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}')
            return
        
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        results = self.model(
            rgb_image, verbose=False, conf=self.conf_threshold, iou=self.iou_threshold)
        
        msg_array = Detection2DArray()
        msg_array.header = ros_img.header
        
        if len(results) == 0 or len(results[0].boxes) == 0:
            if not self.publish_placeholder:
                return
            detection = Detection2D()
            detection.bbox.center.position.x = 99999.0
            detection.bbox.center.position.y = 99999.0
            detection.bbox.size_x = 99999.0
            detection.bbox.size_y = 99999.0
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = '0'
            hypothesis.hypothesis.score = 1.0
            detection.results.append(hypothesis)
            msg_array.detections.append(detection)
        else:
            boxes = results[0].boxes
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                if class_id >= len(self.model.names):
                    continue
                
                class_name = self.model.names[class_id]
                if class_name not in self.target_digits:
                    continue
                
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0
                width = x2 - x1
                height = y2 - y1
                
                detection = Detection2D()
                detection.bbox.center.position.x = float(center_x)
                detection.bbox.center.position.y = float(center_y)
                detection.bbox.size_x = float(width)
                detection.bbox.size_y = float(height)
                
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.hypothesis.class_id = class_name
                hypothesis.hypothesis.score = confidence
                detection.results.append(hypothesis)
                
                msg_array.detections.append(detection)
                
                # 只打印识别到的数字（关键日志）
                self.get_logger().info(f'Detected: {class_name} (conf: {confidence:.2f})')
        
        if len(msg_array.detections) > 0:
            self.detection_pub.publish(msg_array)

def main(args=None):
    rclpy.init(args=args)
    detector = RosYOLODetector()
    try:
        rclpy.spin(detector)
    except KeyboardInterrupt:
        pass
    finally:
        detector.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
