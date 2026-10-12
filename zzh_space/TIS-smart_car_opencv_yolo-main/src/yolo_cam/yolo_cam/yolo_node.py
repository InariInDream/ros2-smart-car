#!/usr/bin/env python3
import sys
import time
from pathlib import Path

import cv2
import rclpy
import torch
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

from yolo_msg.msg import Detection, Detections


class YOLONode(Node):
    def __init__(self):
        super().__init__('yolo_node')
        self.bridge = CvBridge()

        package_dir = Path(__file__).resolve().parent
        default_ultralytics_dir = self._find_default_ultralytics_dir(package_dir)

        self.declare_parameter('ultralytics_dir', str(default_ultralytics_dir))
        self.declare_parameter('model_path', 'best.pt')
        self.declare_parameter('task', 'detect')
        self.declare_parameter('device', 'auto')
        self.declare_parameter('confidence_threshold', 0.4)
        self.declare_parameter('input_topic', 'camera/image_raw')
        self.declare_parameter('detections_topic', 'yolo/detections')
        self.declare_parameter('annotated_image_topic', 'yolo/image')
        self.declare_parameter('publish_annotated_image', True)
        self.declare_parameter('log_fps', True)
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('max_det', 100)
        self.declare_parameter('half', False)

        ultralytics_dir = self._resolve_ultralytics_dir(default_ultralytics_dir)
        self._setup_ultralytics_paths(ultralytics_dir)
        from ultralytics import YOLO

        self.task = self.get_parameter('task').get_parameter_value().string_value
        self.device = self._resolve_device()
        self.confidence_threshold = (
            self.get_parameter('confidence_threshold').get_parameter_value().double_value
        )
        self.input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        self.detections_topic = (
            self.get_parameter('detections_topic').get_parameter_value().string_value
        )
        self.annotated_image_topic = (
            self.get_parameter('annotated_image_topic').get_parameter_value().string_value
        )
        self.publish_annotated_image = (
            self.get_parameter('publish_annotated_image').get_parameter_value().bool_value
        )
        self.log_fps = self.get_parameter('log_fps').get_parameter_value().bool_value
        self.imgsz = self.get_parameter('imgsz').get_parameter_value().integer_value
        self.max_det = self.get_parameter('max_det').get_parameter_value().integer_value
        self.half = self.get_parameter('half').get_parameter_value().bool_value
        if self.device == 'cpu' and self.half:
            self.get_logger().warning('CPU 模式下不启用 half，已自动切换为 false')
            self.half = False

        self.subscription = self.create_subscription(
            Image,
            self.input_topic,
            self.image_callback,
            10
        )
        self.result_pub = self.create_publisher(Detections, self.detections_topic, 10)
        self.image_pub = None
        if self.publish_annotated_image:
            self.image_pub = self.create_publisher(Image, self.annotated_image_topic, 10)

        model_path = self._resolve_model_path(ultralytics_dir)
        if not model_path.exists():
            self.get_logger().error(f'模型文件不存在: {model_path}')
            raise FileNotFoundError(model_path)

        self.model = YOLO(str(model_path), task=self.task)
        self.names = self.model.names

        self.start_time = time.time()
        self.frame_count = 0

        self.get_logger().info(
            'YOLOv8 模型加载成功: '
            f'model={model_path}, device={self.device}, '
            f'conf={self.confidence_threshold:.2f}, imgsz={self.imgsz}, '
            f'publish_image={self.publish_annotated_image}'
        )

    def _find_default_ultralytics_dir(self, package_dir: Path) -> Path:
        candidates = [
            package_dir / 'YOLOv8',
            Path.cwd() / 'src' / 'yolo_cam' / 'yolo_cam' / 'YOLOv8',
            Path.cwd() / 'yolo_cam' / 'YOLOv8',
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()

        return (package_dir / 'YOLOv8').resolve()

    def _resolve_ultralytics_dir(self, default_dir: Path) -> Path:
        configured_dir = self.get_parameter('ultralytics_dir').get_parameter_value().string_value
        ultralytics_dir = self._resolve_configured_path(configured_dir, default_dir)
        return ultralytics_dir.resolve()

    def _setup_ultralytics_paths(self, ultralytics_dir: Path) -> None:
        utils_path = ultralytics_dir / 'ultralytics' / 'yolo' / 'utils'
        for candidate in (ultralytics_dir, utils_path):
            candidate_str = str(candidate)
            if candidate.exists() and candidate_str not in sys.path:
                sys.path.append(candidate_str)

    def _resolve_model_path(self, ultralytics_dir: Path) -> Path:
        configured_path = self.get_parameter('model_path').get_parameter_value().string_value
        model_path = Path(configured_path).expanduser()
        if not model_path.is_absolute():
            model_path = ultralytics_dir / model_path
        return model_path.resolve()

    def _resolve_configured_path(self, configured_path: str, default_path: Path) -> Path:
        if not configured_path:
            return default_path

        path = Path(configured_path).expanduser()
        if path.is_absolute():
            return path

        search_roots = []
        for parent in Path(__file__).resolve().parents:
            search_roots.append(parent)
            if parent.name == 'install':
                search_roots.append(parent.parent)

        search_roots.extend([Path.cwd(), Path.cwd().parent])

        for root in search_roots:
            candidate = (root / path).resolve()
            if candidate.exists():
                return candidate

        return (search_roots[0] / path).resolve()

    def _resolve_device(self) -> str:
        configured_device = self.get_parameter('device').get_parameter_value().string_value.strip()
        if configured_device and configured_device != 'auto':
            return configured_device
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    def image_callback(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        annotated_frame = frame.copy() if self.publish_annotated_image else None

        self.frame_count += 1
        current_time = time.time()
        if self.log_fps and current_time - self.start_time >= 1.0:
            fps = self.frame_count / (current_time - self.start_time)
            self.get_logger().info(f'yolo实时帧率: {fps:.2f} FPS')
            self.start_time = current_time
            self.frame_count = 0

        results_list = self.model(
            frame,
            device=self.device,
            conf=self.confidence_threshold,
            imgsz=self.imgsz,
            max_det=self.max_det,
            half=self.half,
            verbose=False
        )

        results = results_list[0]
        boxes = results.boxes
        detection_array = Detections()

        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy()

            for i in range(len(boxes)):
                x1, y1, x2, y2 = xyxy[i]
                conf = float(confs[i])
                cls_id = int(clss[i])
                cls_name = str(self.names.get(cls_id, cls_id))

                det_msg = Detection()
                det_msg.cls = cls_name
                det_msg.conf = conf
                det_msg.xyxy = [float(x1), float(y1), float(x2), float(y2)]
                detection_array.detections.append(det_msg)

                if annotated_frame is not None:
                    label = f'{cls_name} {conf:.2f}'
                    cv2.rectangle(
                        annotated_frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (0, 255, 0),
                        2
                    )
                    cv2.putText(
                        annotated_frame,
                        label,
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

        self.result_pub.publish(detection_array)

        if annotated_frame is not None and self.image_pub is not None:
            out_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
            out_msg.header = msg.header
            self.image_pub.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = YOLONode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

