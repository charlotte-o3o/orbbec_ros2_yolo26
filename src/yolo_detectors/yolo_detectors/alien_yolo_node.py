#!/usr/bin/env python3

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

os.environ["QT_LOGGING_RULES"] = "*.warning=false"

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

from std_msgs.msg import Float32, Int32
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D
import message_filters

from cv_bridge import CvBridge
from ultralytics import YOLO


class AlienYoloNode(Node):

    def __init__(self):
        super().__init__('alien_yolo_node')
        
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
        #                         SUBSCRIBERS (SYNCHRONISÉS)                 #
        # ------------------------------------------------------------------ #
        self.sub_color = message_filters.Subscriber(self, Image, '/orbbec_external/color/image_raw')
        self.sub_depth = message_filters.Subscriber(self, Image, '/orbbec_external/depth/image_raw')

        # Synchronisation temporelle sur les flux Dockerisés
        self.ts = message_filters.TimeSynchronizer([self.sub_color, self.sub_depth], queue_size=10)
        # Fonctions callback associées au subscribers
        # Elles sont appelées dès qu'un message est reçu sur le topic concerné
        self.ts.registerCallback(self._on_frames_received)

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
        self.box_color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
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

        # Vidéo / Photos
        self.video_writer = None
        if self.record_mode: os.makedirs("captures_videos", exist_ok=True)
        if self.save_mode: os.makedirs("captures_img", exist_ok=True)
        self.last_save_time = 0.0

        # ------------------------------------------------------------------ #
        #                          QUEUES & THREADS                          #
        # ------------------------------------------------------------------ #
        self.inference_queue: queue.Queue = queue.Queue(maxsize=1)
        self.display_queue:   queue.Queue = queue.Queue(maxsize=1)

        self.infer_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self.infer_thread.start()

        # Timer indépendant pour gérer l'affichage graphique et les publications ROS2 (~30 FPS)
        self.display_timer = self.create_timer(0.033, self._display_and_publish_loop)

        self.get_logger().info("=== AlienYoloNode démarré ===")
        self.get_logger().info(f"CSV des distances : {self.csv_path}")

    # ------------------------------------------------------------------ #
    #                        CALLBACK RECEPTION FLUX                     #
    # ------------------------------------------------------------------ #
    def _on_frames_received(self, msg_color, msg_depth):
        """Récupère les frames synchronisées et les pousse vers YOLO"""
        try:
            frame_bgr = self.bridge.imgmsg_to_cv2(msg_color, desired_encoding='bgr8')
            raw_depth = self.bridge.imgmsg_to_cv2(msg_depth, desired_encoding='mono16')
            
            height, width, _ = frame_bgr.shape
            local_depth = cv2.resize(raw_depth, (width, height), interpolation=cv2.INTER_NEAREST)

            # Initialisation unique du VideoWriter si nécessaire
            if self.record_mode and self.video_writer is None:
                ts  = time.strftime("%Y-%m-%d_%H-%M-%S")
                vp  = os.path.join("captures_videos", f"capture_{ts}.avi")
                fcc = cv2.VideoWriter_fourcc(*'XVID')
                self.video_writer = cv2.VideoWriter(vp, fcc, 25.0, (width, height))

            # Pousse les images dans la file d'inférence (sans bloquer si elle est pleine)
            if not self.inference_queue.full():
                self.inference_queue.put((frame_bgr.copy(), local_depth.copy(), width, height))
        except Exception as e:
            self.get_logger().error(f"Erreur décodage frames : {e}")

    # ------------------------------------------------------------------ #
    #                  BOUCLE AUTONOME D'AFFICHAGE                       #
    # ------------------------------------------------------------------ #
    def _display_and_publish_loop(self):
        """Gère les fenêtres UI et publie sur les topics ROS2 sans bloquer les callbacks"""
        if not self.display_queue.empty():
            annotated, num_objects, det_array, last_distance = self.display_queue.get_nowait()
            height, width, _ = annotated.shape

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

            # Publication bounding boxes et compteurs
            self.pub_bbox.publish(det_array)
            
            count_msg = Int32()
            count_msg.data = num_objects
            self.pub_count.publish(count_msg)

            # Sauvegarde photo ponctuelle
            if num_objects > 0 and self.save_mode:
                now = time.time()
                if now - self.last_save_time >= 1.0:
                    self.last_save_time = now
                    ts  = time.strftime("%Y-%m-%d_%H-%M-%S")
                    fn  = f"captures_img/detection_{ts}.jpg"
                    cv2.imwrite(fn, annotated)
                    self.get_logger().info(f"{num_objects} objet(s) détecté(s) — image sauvegardée")

            # Affichage de l'IHM
            cv2.imshow("YOLO26 - Alien Plushie Detection", annotated)
            
        # Obligatoire pour laisser OpenCV rafraîchir la fenêtre
        if cv2.waitKey(1) & 0xFF == 27:
            self.get_logger().info("ECHAP pressé — arrêt demandé.")
            self._shutdown_requested = True

    # ------------------------------------------------------------------ #
    #                        THREAD D'INFÉRENCE                          #
    # ------------------------------------------------------------------ #
    def _inference_worker(self):
        while True:
            item = self.inference_queue.get()
            if item is None: break

            frame_bgr, depth_frame, width, height = item

            t0 = time.perf_counter()
            results = list(self.model(frame_bgr, stream=True, conf=self.conf_thresh, verbose=False, classes=[0]))
            inf_ms = (time.perf_counter() - t0) * 1000.0
            fps = 1000.0 / inf_ms if inf_ms > 0 else 0.0

            annotated = frame_bgr.copy()
            boxes = results[0].boxes
            num_objects = len(boxes) if boxes is not None else 0

            with self.frame_lock:
                self.frame_counter += 1
                frame_id = self.frame_counter

            det_array = Detection2DArray()
            det_array.header.stamp = self.get_clock().now().to_msg()
            det_array.header.frame_id = "camera_color_optical_frame"

            last_distance = 0.0

            if boxes is not None and depth_frame is not None and num_objects > 0:
                for box in boxes:
                    class_id = int(box.cls[0])
                    label = self.model.names[class_id]
                    confie = float(box.conf[0]) * 100
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    distance = self._measure_distance(depth_frame, x1, y1, x2, y2, width, height)

                    if distance > 0:
                        self.csv_writer.writerow([frame_id, round(distance, 4)])
                        last_distance = distance
                        text_dist = f"{distance:.2f}m"
                    else:
                        self.csv_writer.writerow([frame_id, math.nan])
                        text_dist = "---"

                    x_center = max(0, min(int((x1+x2)/2), width-1))
                    y_center = max(0, min(int((y1+y2)/2), height-1))
                    custom_label = f"{label} ({confie:.1f}%) : {text_dist}"
                    cv2.rectangle(annotated, (x1,y1), (x2,y2), self.box_color, 2)
                    cv2.putText(annotated, custom_label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.box_color, 2)
                    cv2.circle(annotated, (x_center, y_center), 5, (0,0,255), -1)

                    self._draw_big_distance(annotated, text_dist, width, height)

                    det = Detection2D()
                    det.header = det_array.header
                    bb = BoundingBox2D()
                    bb.center.position.x = float((x1+x2)/2)
                    bb.center.position.y = float((y1+y2)/2)
                    bb.size_x = float(x2-x1)
                    bb.size_y = float(y2-y1)
                    det.bbox = bb
                    det_array.detections.append(det)
            else:
                self.csv_writer.writerow([frame_id, math.nan])

            cv2.putText(annotated, f"Object(s): {num_objects}", (30,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            cv2.putText(annotated, f"Inference: {inf_ms:.1f} ms ({fps:.0f} FPS) | ECHAP pour quitter", (30,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            if not self.display_queue.full():
                self.display_queue.put((annotated, num_objects, det_array, last_distance))

    def _measure_distance(self, depth_frame, x1, y1, x2, y2, width, height) -> float:
        margin_x = int((x2-x1) * 0.35)
        margin_y = int((y2-y1) * 0.35)
        y1_p = max(0, y1 + margin_y)
        y2_p = min(depth_frame.shape[0], y2 - margin_y)
        x1_p = max(0, x1 + margin_x)
        x2_p = min(depth_frame.shape[1], x2 - margin_x)

        patch = depth_frame[y1_p:y2_p, x1_p:x2_p]
        valid = patch[patch > 0]
        if len(valid) == 0: return 0.0

        median_val = float(np.median(valid))
        std_val    = float(np.std(valid))
        filtered   = valid[np.abs(valid - median_val) < std_val]
        distance   = float(np.median(filtered)) / 1000.0 if len(filtered) > 0 else median_val / 1000.0

        if distance > 0 and len(self.distance_history) > 0:
            if abs(distance - self.distance_history[-1]) > self.max_jump:
                distance = self.distance_history[-1]

        if distance > 0:
            self.distance_history.append(distance)
            if len(self.distance_history) > self.max_history: self.distance_history.pop(0)
            distance = float(np.mean(self.distance_history))
        return distance

    def _draw_big_distance(self, frame, text: str, width: int, height: int):
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, th_ = 10.0, 14
        (tw, th), _ = cv2.getTextSize(text, font, scale, th_)
        tx, ty = 30, height - 30
        overlay = frame.copy()
        cv2.rectangle(overlay, (tx-10, ty-th-10), (tx+tw+10, ty+10), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.putText(frame, text, (tx, ty), font, scale, (0,255,0), th_, cv2.LINE_AA)

    def destroy_node(self):
        self.inference_queue.put(None)
        self.csv_file.close()
        self.get_logger().info(f"Distances CSV sauvegardées : {self.csv_path}")
        if self.video_writer is not None:
            self.video_writer.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = AlienYoloNode()
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