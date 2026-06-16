from rclpy.node import Node
import message_filters
from sensor_msgs.msg import Image
import rclpy
import cv2
from cv_bridge import CvBridge
from ultralytics import YOLO

class TestSubscriberNode(Node):

    def __init__(self):
        super().__init__('test_subscriber_node')

        self.get_logger().info("*** Subscriber Node Launched successfully ***")

        # Initialisation du convertisseur CvBridge
        self.bridge = CvBridge()

        self.model = YOLO('alien_plushie_v3.pt')


        self.sub_color = self.create_subscription(
            Image, 
            '/orbbec_external/color/image_raw', 
            self.color_callback, 
            10
        )
        
        self.sub_depth = self.create_subscription(
            Image, 
            '/orbbec_external/depth/image_raw', 
            self.depth_callback, 
            10
        )

    def color_callback(self, msg):
        try:
            # Conversion du message ROS en image OpenCV standard (BGR)
            cv_color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            results = self.model(cv_color_image, verbose=False)

            annotated_image = results[0].plot()

            cv2.imshow("Alien Plushie Detection", annotated_image)

            # Pour qu'OpenCV rafraîchisse la fenêtre et traite les événements
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().info(f"Color conversion error : {e}")

    def depth_callback(self, msg):
        try:
            # Conversion du message ROS en image OpenCV brute
            cv_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

            depth_vis = cv2.normalize(cv_depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

            cv2.imshow("Colorized Depth Image", depth_colormap)

            # Pour qu'OpenCV rafraîchisse la fenêtre et traite les événements
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().info(f"Depth conversion error : {e}")

    def destroy_node(self):
        cv2.destroyAllWindows()
        return super().destroy_node()
    
def main(args=None):
    rclpy.init(args=args)
    node = TestSubscriberNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


