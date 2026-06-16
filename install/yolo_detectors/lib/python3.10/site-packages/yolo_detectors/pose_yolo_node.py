#!/usr/bin/env python3

import os
import contextlib
import threading
import queue
import json
import math
import time

import cv2
import numpy as np


os.environ["QT_LOGGING_RULES"] = "*.warning=false"

import rclpy
from rclpy.node import Node

from std_msgs.msg   import Float32, Int32, String
from sensor_msgs.msg import Image
import message_filters

from cv_bridge import CvBridge
from ultralytics import YOLO

# Noms des 17 keypoints COCO (utilisés pour le JSON publié)
COCO_KEYPOINTS = [
    "nose","left_eye","right_eye","left_ear","right_ear",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_hip","right_hip",
    "left_knee","right_knee","left_ankle","right_ankle"
]


class PoseYoloNode(Node):

    def __init__(self):
        super().__init__('pose_yolo_node')

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
        #                         SUBSCRIBERS (SYNCHRONISÉS)                 #
        # ------------------------------------------------------------------ #
        # Création des abonnements individuels
        self.sub_color = message_filters.Subscriber(self, Image, '/camera/color/image_raw')
        self.sub_depth = message_filters.Subscriber(self, Image, '/camera/depth/image_raw')

        # Synchronisation des flux (les messages doivent avoir exactement le même timestamp dans le header)
        # queue_size=10 pour amortir les légers retards de calcul
        self.ts = message_filters.TimeSynchronizer([self.sub_color, self.sub_depth], queue_size=10)
        
        # Redirection vers la fonction de callback
        self.ts.registerCallback(self._on_frames_received)

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


        self.get_logger().info("=== PoseYoloNode démarré ===")
        if self.save_mode:
            self.get_logger().info("Mode sauvegarde d'images activé.")

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
    def _on_frames_received(self, msg_color, msg_depth):
        """Remplace l'ancienne méthode _camera_loop"""
        
        # 1. Conversion des messages ROS2 en matrices OpenCV / Numpy
        frame_bgr = self.bridge.imgmsg_to_cv2(msg_color, desired_encoding='bgr8')
        raw_depth = self.bridge.imgmsg_to_cv2(msg_depth, desired_encoding='mono16')
        
        height, width, _ = frame_bgr.shape
        
        # 2. Redimensionnement de la profondeur si nécessaire (comme dans vos codes originaux)
        local_depth = cv2.resize(raw_depth, (width, height), interpolation=cv2.INTER_NEAREST)

        # [Optionnel pour Pose Node] Sauvegarde thread-safe du snapshot de profondeur
        if hasattr(self, '_frame_depth_lock'):
            with self._frame_depth_lock:
                self._frame_depth = local_depth

        # 3. Initialisation du VideoWriter (si activé dans alien_detection)
        if hasattr(self, 'record_mode') and self.record_mode and self.video_writer is None:
            ts  = time.strftime("%Y-%m-%d_%H-%M-%S")
            vp  = os.path.join("captures_videos", f"capture_{ts}.avi")
            fcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(vp, fcc, 25.0, (width, height))

        # 4. Envoi au thread de calcul YOLO (votre file d'attente reste inchangée !)
        if not self.inference_queue.full():
            if hasattr(self, '_frame_depth_lock'):
                # Pour le Pose Node
                self.inference_queue.put((frame_bgr.copy(), width, height))
            else:
                # Pour le Alien Node
                self.inference_queue.put((frame_bgr.copy(), local_depth.copy(), width, height))

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
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PoseYoloNode()
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
