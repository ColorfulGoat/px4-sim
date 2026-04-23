import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import time

class YoloDetectionNode(Node):
    def __init__(self):
        super().__init__('yolo_detection_node')
        
        self.bridge = CvBridge()
        self.model = YOLO('yolov8n.pt')
        
        self.subscription = self.create_subscription(
            Image,
            '/world/baylands/model/x500_depth_0/link/camera_link/sensor/IMX214/image',
            self.image_callback,
            10)
        
        self.publisher = self.create_publisher(Image, '/yolo/detection_image', 10)
        
        self.frame_count = 0
        self.start_time = time.time()
        self.get_logger().info('YOLO Detection Node started!')

    def image_callback(self, msg):
        # Convert ROS image to OpenCV handling different encodings
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            if len(cv_image.shape) == 2:
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
            elif cv_image.shape[2] == 4:
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2BGR)

        if cv_image is None or cv_image.size == 0:
            self.get_logger().warn('Empty image received')
            return

        # Run YOLO inference
        t0 = time.time()
        results = self.model(cv_image, verbose=False, conf=0.1)
        latency = (time.time() - t0) * 1000

        # Draw detections
        annotated = results[0].plot()

        # Calculate FPS
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed

        # Overlay stats
        cv2.putText(annotated, f'FPS: {fps:.1f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated, f'Latency: {latency:.1f}ms', (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Log detections
        for box in results[0].boxes:
            cls = self.model.names[int(box.cls)]
            conf = float(box.conf)
            self.get_logger().info(f'Detected: {cls} ({conf:.2f})')

        # Publish annotated image
        out_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        out_msg.header = msg.header
        self.publisher.publish(out_msg)

        self.get_logger().info(f'Latency: {latency:.1f}ms | FPS: {fps:.1f}')

def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
