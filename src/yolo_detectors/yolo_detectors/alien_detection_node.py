#!/usr/bin/env python3

"""
ROS2 Node — YOLO26 Alien Plushie Detection (Orbbec Femto Bolt)
===============================================================
Topics publiés :
  /alien_detection/image_annotated   (sensor_msgs/Image)
  /alien_detection/distance          (std_msgs/Float32)
  /alien_detection/bbox              (vision_msgs/Detection2DArray)
  /alien_detection/num_objects       (std_msgs/Int32)

Paramètres ROS2 déclarés :
  model_path          (str,   default: "alien_plushie_v3.pt")
  confidence          (float, default: 0.50)
  save_mode           (bool,  default: False)
  record_mode         (bool,  default: True)
  max_history         (int,   default: 5)
  max_jump            (float, default: 0.5)
"""

import os
import sys
import contextlib
import threading
import queue
import csv
import math
import time
import random

import cv2
import numpy as np

# --- Import propre Orbbec SDK
with contextlib.redirect_stdout(None):
    from pyorbbecsdk import Pipeline, Config, OBSensorType, OBAlignMode

os.environ["QT_LOGGING_RULES"] = "*.warning=false"

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

from std_msgs.msg import Float32, Int32
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D
from geometry_msgs.msg import Pose2D

from cv_bridge import CvBridge
from ultralytics import YOLO


class AlienDetectionNode(Node):

    def __init__(self):
        super().__init__('alien_detection_node')
        
        self._shutdown_requested = False

        # ------------------------------------------------------------------ #
        #                          PARAMETRES ROS2                           #
        # ------------------------------------------------------------------ #
        self.declare_parameter('model_path',  'alien_plushie_v3.pt')
        self.declare_parameter('confidence',  0.50)
        self.declare_parameter('save_mode',   False)
        self.declare_parameter('record_mode', True)
        self.declare_parameter('max_history', 5)
        self.declare_parameter('max_jump',    0.5)

        self.model_path  = self.get_parameter('model_path').value
        self.conf_thresh = self.get_parameter('confidence').value
        self.save_mode   = self.get_parameter('save_mode').value
        self.record_mode = self.get_parameter('record_mode').value
        self.max_history = self.get_parameter('max_history').value
        self.max_jump    = self.get_parameter('max_jump').value

        # ------------------------------------------------------------------ #
        #                           PUBLISHERS                               #
        # ------------------------------------------------------------------ #
        self.pub_image    = self.create_publisher(Image,            '/alien_detection/image_annotated', 10)
        self.pub_distance = self.create_publisher(Float32,          '/alien_detection/distance',        10)
        self.pub_bbox     = self.create_publisher(Detection2DArray, '/alien_detection/bbox',            10)
        self.pub_count    = self.create_publisher(Int32,            '/alien_detection/num_objects',     10)

        # ------------------------------------------------------------------ #
        #                         INITIALISATION                             #
        # ------------------------------------------------------------------ #
        self.bridge = CvBridge()

        # Couleur aléatoire de la bounding box
        self.box_color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))

        # Historique de distance pour le lissage
        self.distance_history: list[float] = []

        # YOLO
        self.get_logger().info(f"Chargement du modèle : {self.model_path}")
        self.model = YOLO(self.model_path)
        self.get_logger().info("Modèle chargé.")

        # CSV
        log_dir = "distance_logs"
        os.makedirs(log_dir, exist_ok=True)
        ts = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.csv_path = os.path.join(log_dir, f"distances_{ts}.csv")
        self.csv_file   = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["frame_id", "distance_m"])
        self.frame_counter = 0
        self.frame_lock    = threading.Lock()

        # Vidéo
        self.video_writer = None
        if self.record_mode:
            os.makedirs("captures_videos", exist_ok=True)

        # Photos
        if self.save_mode:
            os.makedirs("captures_img", exist_ok=True)
        self.last_save_time = 0.0

        # ------------------------------------------------------------------ #
        #                          QUEUES & THREAD                           #
        # ------------------------------------------------------------------ #
        self.inference_queue: queue.Queue = queue.Queue(maxsize=1)
        self.display_queue:   queue.Queue = queue.Queue(maxsize=1)

        self.infer_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self.infer_thread.start()

        # ------------------------------------------------------------------ #
        #                         PIPELINE ORBBEC                           #
        # ------------------------------------------------------------------ #
        self.pipe   = Pipeline()
        self.config = Config()
        self._start_camera()

        # Timer principal (lecture caméra + publication)
        self.timer = self.create_timer(0.001, self._camera_loop)

        self.get_logger().info("=== AlienDetectionNode démarré ===")
        self.get_logger().info(f"CSV des distances : {self.csv_path}")
        if self.record_mode:
            self.get_logger().info("Mode enregistrement vidéo activé.")
        if self.save_mode:
            self.get_logger().info("Mode sauvegarde d'images activé.")

    # ------------------------------------------------------------------ #
    #                        DÉMARRAGE CAMÉRA                            #
    # ------------------------------------------------------------------ #
    def _start_camera(self):
        profile_list  = self.pipe.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = profile_list.get_default_video_stream_profile()
        self.config.enable_stream(color_profile)

        profile_list  = self.pipe.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile = profile_list.get_default_video_stream_profile()
        self.config.enable_stream(depth_profile)

        try:
            self.config.set_align_mode(OBAlignMode.SW_MODE)
        except Exception as e:
            self.get_logger().warn(f"Alignement SW non supporté : {e}")

        self.pipe.start(self.config)

    # ------------------------------------------------------------------ #
    #                        THREAD D'INFÉRENCE                          #
    # ------------------------------------------------------------------ #
    def _inference_worker(self):
        while True:
            item = self.inference_queue.get()
            if item is None:
                break

            frame_bgr, depth_frame, width, height = item

            # ---- Inférence YOLO ----
            t0      = time.perf_counter()
            results = list(self.model(frame_bgr, stream=True,
                                      conf=self.conf_thresh,
                                      verbose=False, classes=[0]))
            inf_ms  = (time.perf_counter() - t0) * 1000.0
            fps     = 1000.0 / inf_ms if inf_ms > 0 else 0.0

            annotated  = frame_bgr.copy()
            boxes      = results[0].boxes
            num_objects = len(boxes) if boxes is not None else 0

            # ---- Compteur de frame ----
            with self.frame_lock:
                self.frame_counter += 1
                frame_id = self.frame_counter

            # ---- Detection2DArray ----
            det_array      = Detection2DArray()
            det_array.header.stamp    = self.get_clock().now().to_msg()
            det_array.header.frame_id = "camera_color_optical_frame"

            last_distance = 0.0  # distance du dernier objet (pour publication Float32)

            if boxes is not None and depth_frame is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    label    = self.model.names[class_id]
                    confie   = float(box.conf[0]) * 100
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # ---- Mesure de distance robuste ----
                    distance = self._measure_distance(depth_frame, x1, y1, x2, y2, width, height)

                    # ---- CSV ----
                    if distance > 0:
                        self.csv_writer.writerow([frame_id, round(distance, 4)])
                        last_distance = distance
                        text_dist = f"{distance:.2f}m"
                    else:
                        self.csv_writer.writerow([frame_id, math.nan])
                        text_dist = "---"

                    # ---- Annotation visuelle ----
                    x_center = max(0, min(int((x1+x2)/2), width-1))
                    y_center = max(0, min(int((y1+y2)/2), height-1))
                    custom_label = f"{label} ({confie:.1f}%) : {text_dist}"
                    cv2.rectangle(annotated, (x1,y1), (x2,y2), self.box_color, 2)
                    cv2.putText(annotated, custom_label, (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.box_color, 2)
                    cv2.circle(annotated, (x_center, y_center), 5, (0,0,255), -1)

                    # ---- Grande distance bas-gauche ----
                    self._draw_big_distance(annotated, text_dist, width, height)

                    # ---- Detection2D message ----
                    det      = Detection2D()
                    det.header = det_array.header
                    bb       = BoundingBox2D()
                    bb.center.position.x = float((x1+x2)/2)
                    bb.center.position.y = float((y1+y2)/2)
                    bb.size_x = float(x2-x1)
                    bb.size_y = float(y2-y1)
                    det.bbox  = bb
                    det_array.detections.append(det)

            else:
                # Aucune détection cette frame
                self.csv_writer.writerow([frame_id, math.nan])

            # ---- HUD ----
            cv2.putText(annotated, f"Object(s): {num_objects}", (30,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            cv2.putText(annotated,
                        f"Inference: {inf_ms:.1f} ms ({fps:.0f} FPS) | ECHAP pour quitter",
                        (30,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            if not self.display_queue.full():
                self.display_queue.put((annotated, num_objects, det_array, last_distance))

    # ------------------------------------------------------------------ #
    #                        MESURE DE DISTANCE                          #
    # ------------------------------------------------------------------ #
    def _measure_distance(self, depth_frame, x1, y1, x2, y2, width, height) -> float:
        margin_x = int((x2-x1) * 0.35)
        margin_y = int((y2-y1) * 0.35)
        y1_p = max(0, y1 + margin_y)
        y2_p = min(depth_frame.shape[0], y2 - margin_y)
        x1_p = max(0, x1 + margin_x)
        x2_p = min(depth_frame.shape[1], x2 - margin_x)

        patch = depth_frame[y1_p:y2_p, x1_p:x2_p]
        valid = patch[patch > 0]

        if len(valid) == 0:
            return 0.0

        median_val = float(np.median(valid))
        std_val    = float(np.std(valid))
        filtered   = valid[np.abs(valid - median_val) < std_val]
        distance   = float(np.median(filtered)) / 1000.0 if len(filtered) > 0 else median_val / 1000.0

        # Filtre saut aberrant
        if distance > 0 and len(self.distance_history) > 0:
            if abs(distance - self.distance_history[-1]) > self.max_jump:
                distance = self.distance_history[-1]

        # Lissage par moyenne glissante
        if distance > 0:
            self.distance_history.append(distance)
            if len(self.distance_history) > self.max_history:
                self.distance_history.pop(0)
            distance = float(np.mean(self.distance_history))

        return distance

    # ------------------------------------------------------------------ #
    #                     AFFICHAGE GRANDE DISTANCE                      #
    # ------------------------------------------------------------------ #
    def _draw_big_distance(self, frame, text: str, width: int, height: int):
        font       = cv2.FONT_HERSHEY_SIMPLEX
        scale, th_ = 10.0, 14
        (tw, th), _ = cv2.getTextSize(text, font, scale, th_)
        tx, ty = 30, height - 30
        overlay = frame.copy()
        cv2.rectangle(overlay, (tx-10, ty-th-10), (tx+tw+10, ty+10), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.putText(frame, text, (tx, ty), font, scale, (0,255,0), th_, cv2.LINE_AA)

    # ------------------------------------------------------------------ #
    #                        BOUCLE PRINCIPALE                           #
    # ------------------------------------------------------------------ #
    def _camera_loop(self):
        # ---- Lecture caméra ----
        frames = self.pipe.wait_for_frames(100)
        if frames is None:
            return

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if color_frame is None or depth_frame is None:
            return

        data    = color_frame.get_data()
        enc_img = np.frombuffer(data, dtype=np.uint8)
        frame_bgr = cv2.imdecode(enc_img, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            return

        height, width, _ = frame_bgr.shape

        # Profondeur
        depth_data = depth_frame.get_data()
        raw_depth  = np.frombuffer(depth_data, dtype=np.uint16).reshape(
            (depth_frame.get_height(), depth_frame.get_width()))
        local_depth = cv2.resize(raw_depth, (width, height), interpolation=cv2.INTER_NEAREST)

        # Init VideoWriter
        if self.record_mode and self.video_writer is None:
            ts  = time.strftime("%Y-%m-%d_%H-%M-%S")
            vp  = os.path.join("captures_videos", f"capture_{ts}.avi")
            fcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(vp, fcc, 25.0, (width, height))
            self.get_logger().info(f"Fichier vidéo créé : {vp}")

        # Envoi au thread YOLO
        if not self.inference_queue.full():
            self.inference_queue.put((frame_bgr.copy(), local_depth.copy(), width, height))

        # ---- Récupération résultats et publication ----
        if not self.display_queue.empty():
            annotated, num_objects, det_array, last_distance = self.display_queue.get_nowait()

            # Date/heure sur l'image
            dt_str = time.strftime("%d/%m/%Y  %H:%M:%S")
            cv2.putText(annotated, dt_str, (width-225, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

            # Enregistrement vidéo
            if self.record_mode and self.video_writer is not None:
                self.video_writer.write(annotated)

            # Publication image annotée
            ros_img = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            ros_img.header.stamp    = self.get_clock().now().to_msg()
            ros_img.header.frame_id = "camera_color_optical_frame"
            self.pub_image.publish(ros_img)

            # Publication distance
            dist_msg = Float32()
            dist_msg.data = float(last_distance) if last_distance > 0 else float('nan')
            self.pub_distance.publish(dist_msg)

            # Publication bounding boxes
            self.pub_bbox.publish(det_array)

            # Publication nombre d'objets
            count_msg = Int32()
            count_msg.data = num_objects
            self.pub_count.publish(count_msg)

            # Sauvegarde photo
            if num_objects > 0 and self.save_mode:
                now = time.time()
                if now - self.last_save_time >= 1.0:
                    self.last_save_time = now
                    ts  = time.strftime("%Y-%m-%d_%H-%M-%S")
                    fn  = f"captures_img/detection_{ts}.jpg"
                    cv2.imwrite(fn, annotated)
                    self.get_logger().info(f"{num_objects} objet(s) détecté(s) — image sauvegardée")

            # Affichage local
            cv2.imshow("YOLO26 - Alien Plushie Detection", annotated)
            if cv2.waitKey(1) & 0xFF == 27:
                self.get_logger().info("ECHAP pressé — arrêt demandé.")
                self._shutdown_requested = True

    # ------------------------------------------------------------------ #
    #                          NETTOYAGE                                 #
    # ------------------------------------------------------------------ #
    def destroy_node(self):
        self.inference_queue.put(None)
        self.csv_file.close()
        self.get_logger().info(f"Distances CSV sauvegardées : {self.csv_path}")
        if self.video_writer is not None:
            self.video_writer.release()
            self.get_logger().info("Enregistrement vidéo finalisé.")
        try:
            self.pipe.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = AlienDetectionNode()
    try:
        while rclpy.ok() and not node._shutdown_requested:
            rclpy.spin_once(node, timeout_sec=0.01)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    os._exit(0)

if __name__ == '__main__':
    main()
