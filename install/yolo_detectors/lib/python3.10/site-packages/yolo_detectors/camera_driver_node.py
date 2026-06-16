#!/usr/bin/env python3
import os
import contextlib
import cv2
import threading
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

with contextlib.redirect_stdout(None):
    from pyorbbecsdk import Pipeline, Config, OBSensorType, OBAlignMode

class CameraDriverNode(Node):
    def __init__(self):
        super().__init__('camera_driver_node')
        self.bridge = CvBridge()
        self.running = True
        
        # Publishers pour les flux bruts/alignés
        self.pub_color = self.create_publisher(Image, '/camera/color/image_raw', 10)
        self.pub_depth = self.create_publisher(Image, '/camera/depth/image_aligned', 10)

        # Config Orbbec
        self.pipe = Pipeline()
        self.config = Config()
        self._start_camera()

        self.capture_thread = threading.Thread(target=self._camera_thread_loop, daemon=True)
        self.capture_thread.start()

        self.get_logger().info("=== CameraDriverNode Started ===")

    def _start_camera(self):
        # 1. Recherche d'un profil couleur adapté (1280x720 en BGR ou MJPEG)
        color_profiles = self.pipe.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = None
        try:
            # On cherche de préférence du 1280x720 à 30 FPS
            color_profile = color_profiles.get_video_stream_profile(1280, 720, None, 30)
        except:
            color_profile = color_profiles.get_default_video_stream_profile()

        # 2. Recherche d'un profil profondeur adapté (généralement 640x576 ou 640x480)
        depth_profiles = self.pipe.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile = None
        try:
            # Profil ToF standard
            depth_profile = depth_profiles.get_video_stream_profile(640, 576, None, 30)
        except:
            depth_profile = depth_profiles.get_default_video_stream_profile()

        self.get_logger().info(f"Selected color profile : {color_profile.get_width()}x{color_profile.get_height()}")
        self.get_logger().info(f"Selected depth profile : {depth_profile.get_width()}x{depth_profile.get_height()}")

        # 3. Activation des flux
        self.config.enable_stream(color_profile)
        self.config.enable_stream(depth_profile)
        
        # 4. Activation de l'alignement
        try:
            self.config.set_align_mode(OBAlignMode.SW_MODE)
            self.get_logger().info("Software alignment activation successful")
        except Exception as e:
            self.get_logger().warn(f"Alignement unsupported : {e}")

        # 5. Activation de la synchronisation matérielle/SDK
        """try:
            self.pipe.enable_frame_sync()
            self.get_logger().info("Frame synchronization activated")
        except Exception as e:
            self.get_logger().warn(f"Failed to activate frame synchronization : {e}")"""

        self.pipe.start(self.config)

    def _camera_thread_loop(self):
        """Boucle exécutée dans un thread séparé pour ne pas bloquer ROS2"""
        while rclpy.ok() and self.running:
            try:
                # Timeout augmenté à 500ms pour éviter les expirations cycliques
                frames = self.pipe.wait_for_frames(100)
                if not frames:
                    self.get_logger().warn("Timeout : failed to get frame from camera")
                    continue

                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                
                if color_frame is None:
                    self.get_logger().warn("Color frame missing")
                    continue
                if depth_frame is None:
                    self.get_logger().warn("Depth frame missing")
                    continue

                # Traitement Couleur
                color_data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
                frame_bgr = cv2.imdecode(color_data, cv2.IMREAD_COLOR)
                
                # Traitement Profondeur
                depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(
                    (depth_frame.get_height(), depth_frame.get_width()))

                if frame_bgr is None or depth_data is None:
                    continue

                # Synchronisation temporelle stricte
                now = self.get_clock().now().to_msg()

                # Publication Color
                msg_color = self.bridge.cv2_to_imgmsg(frame_bgr, encoding='bgr8')
                msg_color.header.stamp = now
                msg_color.header.frame_id = "camera_link"
                self.pub_color.publish(msg_color)

                # Publication Depth
                msg_depth = self.bridge.cv2_to_imgmsg(depth_data, encoding='mono16')
                msg_depth.header.stamp = now
                msg_depth.header.frame_id = "camera_link"
                self.pub_depth.publish(msg_depth)

            except Exception as e:
                self.get_logger().error(f"Erreur dans la boucle de capture : {e}")

    def destroy_node(self):
        self.running = False
        try: self.pipe.stop()
        except: pass
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = CameraDriverNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()