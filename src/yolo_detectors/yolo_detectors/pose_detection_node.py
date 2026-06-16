#!/usr/bin/env python3

"""
ROS2 Node — YOLO26-Pose Person Detection (Orbbec Femto Bolt)
=============================================================
Topics publiés :
  /pose_detection/image_annotated   (sensor_msgs/Image)
  /pose_detection/distance          (std_msgs/Float32)
  /pose_detection/keypoints         (std_msgs/String  — JSON)
  /pose_detection/num_persons       (std_msgs/Int32)

Paramètres ROS2 déclarés :
  model_path    (str,   default: "yolo26n-pose.pt")
  confidence    (float, default: 0.50)
  save_mode     (bool,  default: False)
"""

import os
import contextlib
import threading
import queue
import json
import math
import time

import cv2
import numpy as np

with contextlib.redirect_stdout(None):
    from pyorbbecsdk import Pipeline, Config, OBSensorType, OBAlignMode

os.environ["QT_LOGGING_RULES"] = "*.warning=false"

import rclpy
from rclpy.node import Node

from std_msgs.msg   import Float32, Int32, String
from sensor_msgs.msg import Image

from cv_bridge import CvBridge
from ultralytics import YOLO

# Noms des 17 keypoints COCO (utilisés pour le JSON publié)
COCO_KEYPOINTS = [
    "nose","left_eye","right_eye","left_ear","right_ear",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_hip","right_hip",
    "left_knee","right_knee","left_ankle","right_ankle"
]


class PoseDetectionNode(Node):

    def __init__(self):
        super().__init__('pose_detection_node')

        self._shutdown_requested = False

        # ------------------------------------------------------------------ #
        #                          PARAMETRES ROS2                           #
        # ------------------------------------------------------------------ #
        self.declare_parameter('model_path', 'yolo26n-pose.pt')
        self.declare_parameter('confidence', 0.50)
        self.declare_parameter('save_mode',  False)

        self.model_path  = self.get_parameter('model_path').value
        self.conf_thresh = self.get_parameter('confidence').value
        self.save_mode   = self.get_parameter('save_mode').value

        # ------------------------------------------------------------------ #
        #                           PUBLISHERS                               #
        # ------------------------------------------------------------------ #
        self.pub_image    = self.create_publisher(Image,   '/pose_detection/image_annotated', 10)
        self.pub_distance = self.create_publisher(Float32, '/pose_detection/distance',        10)
        self.pub_kpts     = self.create_publisher(String,  '/pose_detection/keypoints',       10)
        self.pub_count    = self.create_publisher(Int32,   '/pose_detection/num_persons',     10)

        # ------------------------------------------------------------------ #
        #                         INITIALISATION                             #
        # ------------------------------------------------------------------ #
        self.bridge = CvBridge()

        self.get_logger().info(f"Chargement du modèle : {self.model_path}")
        self.model = YOLO(self.model_path)
        self.get_logger().info("Modèle chargé.")

        if self.save_mode:
            os.makedirs("captures_img", exist_ok=True)
        self.last_save_time = 0.0

        # ------------------------------------------------------------------ #
        #                          QUEUES & THREAD                           #
        # ------------------------------------------------------------------ #
        self.inference_queue: queue.Queue = queue.Queue(maxsize=1)
        self.display_queue:   queue.Queue = queue.Queue(maxsize=1)

        # frame_depth partagé entre thread caméra et thread YOLO
        self._frame_depth      = None
        self._frame_depth_lock = threading.Lock()

        self.infer_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self.infer_thread.start()

        # ------------------------------------------------------------------ #
        #                         PIPELINE ORBBEC                            #
        # ------------------------------------------------------------------ #
        self.pipe   = Pipeline()
        self.config = Config()
        self._start_camera()

        self.timer = self.create_timer(0.001, self._camera_loop)

        self.get_logger().info("=== PoseDetectionNode démarré ===")
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

            frame_bgr, width, height = item

            # Snapshot thread-safe de la frame de profondeur
            with self._frame_depth_lock:
                depth_snap = self._frame_depth.copy() if self._frame_depth is not None else None

            # ---- Inférence YOLO Pose ----
            t0      = time.perf_counter()
            results = list(self.model(frame_bgr, conf=self.conf_thresh, verbose=False))
            inf_ms  = (time.perf_counter() - t0) * 1000.0
            fps     = 1000.0 / inf_ms if inf_ms > 0 else 0.0

            # Annotation YOLO par défaut (squelette complet)
            annotated         = results[0].plot(labels=False)
            boxes             = results[0].boxes
            keypoints_object  = results[0].keypoints
            num_persons       = len(boxes) if boxes is not None else 0

            last_distance = float('nan')
            all_persons_kpts = []  # liste pour le JSON

            if boxes is not None and keypoints_object is not None and depth_snap is not None:
                kpts = keypoints_object.data.cpu().numpy()

                for i, box in enumerate(boxes):
                    if i >= len(kpts):
                        continue

                    person_kpts = kpts[i]

                    # ---- Distance via centre des épaules ----
                    x_ls, y_ls, c_ls = person_kpts[5]
                    x_rs, y_rs, c_rs = person_kpts[6]

                    cx = int((x_ls + x_rs) / 2)
                    cy = int((y_ls + y_rs) / 2)

                    h_d, w_d = depth_snap.shape
                    cx = max(0, min(cx, w_d - 1))
                    cy = max(0, min(cy, h_d - 1))

                    if cx == 0 and cy == 0:
                        dist_m = float('nan')
                        text_dist = "Dist. inconnue"
                    else:
                        raw_dist = depth_snap[cy, cx] / 1000.0
                        dist_m   = raw_dist if raw_dist > 0 else float('nan')
                        text_dist = f"{dist_m:.2f}m" if raw_dist > 0 else "Dist. inconnue"
                        last_distance = dist_m

                    # ---- Annotation visuelle ----
                    cv2.circle(annotated, (cx, cy), 5, (0,0,255), -1)
                    cv2.putText(annotated, text_dist, (cx+10, cy-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

                    # ---- Grande distance bas-gauche (première personne) ----
                    if i == 0:
                        self._draw_big_distance(annotated, text_dist, width, height)

                    # ---- Construction du dict keypoints pour JSON ----
                    person_dict = {"person_id": i, "distance_m": None if math.isnan(dist_m) else dist_m, "keypoints": {}}
                    for k_idx, k_name in enumerate(COCO_KEYPOINTS):
                        if k_idx < len(person_kpts):
                            kx, ky, kc = person_kpts[k_idx]
                            person_dict["keypoints"][k_name] = {
                                "x": float(kx), "y": float(ky), "confidence": float(kc)
                            }
                    all_persons_kpts.append(person_dict)

            # ---- HUD ----
            cv2.putText(annotated, f"Person(s): {num_persons}", (30,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            cv2.putText(annotated,
                        f"Inference: {inf_ms:.1f} ms ({fps:.0f} FPS) | ECHAP pour quitter",
                        (30,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            if not self.display_queue.full():
                self.display_queue.put((annotated, num_persons, last_distance, all_persons_kpts))

    # ------------------------------------------------------------------ #
    #                     AFFICHAGE GRANDE DISTANCE                      #
    # ------------------------------------------------------------------ #
    def _draw_big_distance(self, frame, text: str, width: int, height: int):
        font       = cv2.FONT_HERSHEY_SIMPLEX
        scale, th_ = 4.0, 8
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
        frames = self.pipe.wait_for_frames(100)
        if frames is None:
            return

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if color_frame is None or depth_frame is None:
            return

        # Décodage couleur
        data      = color_frame.get_data()
        enc_img   = np.frombuffer(data, dtype=np.uint8)
        frame_bgr = cv2.imdecode(enc_img, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            return

        height, width, _ = frame_bgr.shape

        # Profondeur (mise à jour du snapshot partagé)
        depth_data = depth_frame.get_data()
        raw_depth  = np.frombuffer(depth_data, dtype=np.uint16).reshape(
            (depth_frame.get_height(), depth_frame.get_width()))
        resized_depth = cv2.resize(raw_depth, (width, height), interpolation=cv2.INTER_NEAREST)
        with self._frame_depth_lock:
            self._frame_depth = resized_depth

        # Envoi au thread YOLO
        if not self.inference_queue.full():
            self.inference_queue.put((frame_bgr.copy(), width, height))

        # ---- Récupération résultats et publication ----
        if not self.display_queue.empty():
            annotated, num_persons, last_distance, kpts_list = self.display_queue.get_nowait()

            # Date/heure
            dt_str = time.strftime("%d/%m/%Y  %H:%M:%S")
            cv2.putText(annotated, dt_str, (width-225, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

            # Publication image annotée
            ros_img = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            ros_img.header.stamp    = self.get_clock().now().to_msg()
            ros_img.header.frame_id = "camera_color_optical_frame"
            self.pub_image.publish(ros_img)

            # Publication distance
            dist_msg      = Float32()
            dist_msg.data = float(last_distance)
            self.pub_distance.publish(dist_msg)

            # Publication keypoints (JSON)
            kpts_msg      = String()
            kpts_msg.data = json.dumps(kpts_list)
            self.pub_kpts.publish(kpts_msg)

            # Publication nombre de personnes
            count_msg      = Int32()
            count_msg.data = num_persons
            self.pub_count.publish(count_msg)

            # Sauvegarde photo
            if num_persons > 0 and self.save_mode:
                now = time.time()
                if now - self.last_save_time >= 0.5:
                    self.last_save_time = now
                    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
                    fn = f"captures_img/pose_{ts}.jpg"
                    cv2.imwrite(fn, annotated)
                    self.get_logger().info(f"{num_persons} personne(s) détectée(s) — image sauvegardée")

            # Affichage local
            cv2.imshow("YOLO26-Pose", annotated)
            if cv2.waitKey(1) & 0xFF == 27:
                self.get_logger().info("ECHAP pressé — arrêt demandé.")
                self._shutdown_requested = True

    # ------------------------------------------------------------------ #
    #                          NETTOYAGE                                 #
    # ------------------------------------------------------------------ #
    def destroy_node(self):
        self.inference_queue.put(None)
        try:
            self.pipe.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PoseDetectionNode()
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
