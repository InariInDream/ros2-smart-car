#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import cv2

class ImageCompressor(Node):
    def __init__(self):
        super().__init__('image_compressor')
        self.bridge = CvBridge()
        
        # 订阅调试图
        self.sub = self.create_subscription(
            Image, '/line_follower/debug_image',
            self.callback, 10)
        
        # 发布压缩图像
        self.pub = self.create_publisher(
            CompressedImage, '/line_follower/debug_image/compressed', 10)
        
        self.get_logger().info('Image compressor started')
    
    def callback(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            
            # 压缩为JPEG
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            result, encoded_img = cv2.imencode('.jpg', cv_img, encode_param)
            
            if result:
                compressed_msg = CompressedImage()
                compressed_msg.header = msg.header
                compressed_msg.format = 'jpeg'
                compressed_msg.data = encoded_img.tobytes()
                self.pub.publish(compressed_msg)
        except Exception as e:
            self.get_logger().error(f'Error: {e}')

def main():
    rclpy.init()
    node = ImageCompressor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
