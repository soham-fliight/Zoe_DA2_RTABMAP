#!/usr/bin/env python3
import os
import sys
import cv2
import numpy as np
import torch
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

# Depth Anything v2 (repo API)
# repo: https://github.com/DepthAnything/Depth-Anything-V2
from depth_anything_v2.dpt import DepthAnythingV2

MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
}

def pick_device(pref: str) -> str:
    pref = (pref or "").lower()
    if pref == "cuda" and torch.cuda.is_available():
        return "cuda"
    if pref == "mps" and torch.backends.mps.is_available():
        return "mps"
    if pref in ("auto", "", "gpu"):
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    return "cpu"


class DA2RGBD(Node):
    def __init__(self):
        super().__init__("da2_rgbd")

        # ---- parameters ----
        self.declare_parameter("video_source", "/home/fliight/mp4_rgbd_ws/video.mp4")
        self.declare_parameter("frame_id", "camera_color_optical_frame")
        self.declare_parameter("resize_width", 640)
        self.declare_parameter("resize_height", 360)
        self.declare_parameter("encoder", "vitl")  # vits | vitb | vitl | vitg
        self.declare_parameter("checkpoint", "/home/fliight/mp4_rgbd_ws/Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth")
        self.declare_parameter("device", "cpu")    # cpu | cuda | mps | auto
        self.declare_parameter("debug_view", False)

        video_path = self.get_parameter("video_source").get_parameter_value().string_value
        self.frame_id = self.get_parameter("frame_id").get_parameter_value().string_value
        self.target_w = int(self.get_parameter("resize_width").get_parameter_value().integer_value)
        self.target_h = int(self.get_parameter("resize_height").get_parameter_value().integer_value)
        encoder = self.get_parameter("encoder").get_parameter_value().string_value
        ckpt = self.get_parameter("checkpoint").get_parameter_value().string_value
        device_pref = self.get_parameter("device").get_parameter_value().string_value
        self.debug_view = bool(self.get_parameter("debug_view").get_parameter_value().bool_value)

        # ---- video ----
        self.get_logger().info(f"Opening video file: {video_path}")
        self.cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {video_path}")
        backend = getattr(self.cap, "getBackendName", lambda: "unknown")()
        self.get_logger().info(f"VideoCapture opened: backend={backend}")

        # ---- model ----
        if encoder not in MODEL_CONFIGS:
            raise RuntimeError(f"Unknown encoder '{encoder}'. Use one of {list(MODEL_CONFIGS.keys())}")
        self.device = pick_device(device_pref)
        self.get_logger().info(f"Loading DepthAnythingV2({encoder}) on {self.device}...")
        self.model = DepthAnythingV2(**MODEL_CONFIGS[encoder])
        if not ckpt or not os.path.isfile(ckpt):
            raise RuntimeError(
                f"Checkpoint not found: '{ckpt}'. Expected .../checkpoints/depth_anything_v2_{encoder}.pth"
            )
        state = torch.load(ckpt, map_location="cpu")
        self.model.load_state_dict(state, strict=False)
        self.model = self.model.to(self.device).eval()
        self.get_logger().info("DepthAnythingV2 loaded OK.")

        # ---- cv bridge ----
        self.bridge = CvBridge()

        # ---- publishers (RELIABLE QoS for depth_image_proc compatibility) ----
        qos_rel = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST
        )
        self.pub_rgb   = self.create_publisher(Image,      '/camera/image_raw',   qos_rel)
        self.pub_depth = self.create_publisher(Image,      '/camera/depth/image', qos_rel)
        self.pub_info  = self.create_publisher(CameraInfo, '/camera/camera_info', qos_rel)

        # ~20 Hz target; actual depends on model/device
        self.timer = self.create_timer(0.05, self.timer_callback)

    def timer_callback(self):
        ret, frame_bgr = self.cap.read()
        if not ret:
            self.get_logger().info("End of video, restarting...")
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return

        if self.target_w > 0 and self.target_h > 0:
            frame_bgr = cv2.resize(frame_bgr, (self.target_w, self.target_h), interpolation=cv2.INTER_AREA)

        # Depth Anything uses OpenCV BGR in their examples
        with torch.inference_mode():
            depth = self.model.infer_image(frame_bgr)
            depth = np.asarray(depth, dtype=np.float32)  # HxW float32

        # Shared timestamp
        stamp = self.get_clock().now().to_msg()

        # Publish RGB (rgb8)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        msg_rgb = self.bridge.cv2_to_imgmsg(rgb, encoding="rgb8")
        msg_rgb.header.frame_id = self.frame_id
        msg_rgb.header.stamp = stamp
        self.pub_rgb.publish(msg_rgb)

        # Publish Depth (32FC1)
        msg_depth = self.bridge.cv2_to_imgmsg(depth, encoding="32FC1")
        msg_depth.header.frame_id = self.frame_id
        msg_depth.header.stamp = stamp
        self.pub_depth.publish(msg_depth)

        # Camera intrinsics (float everything; include K, R, P, D)
        info = CameraInfo()
        info.header = msg_rgb.header
        info.width  = int(rgb.shape[1])
        info.height = int(rgb.shape[0])

        fx_base = 525.0
        fx = fy = float(fx_base * (info.width / 640.0))
        cx = float(info.width) / 2.0
        cy = float(info.height) / 2.0

        info.k = [fx, 0.0, cx,
                  0.0, fy, cy,
                  0.0, 0.0, 1.0]
        # Identity rectification
        info.r = [1.0, 0.0, 0.0,
                  0.0, 1.0, 0.0,
                  0.0, 0.0, 1.0]
        # Projection (no Tx/Ty)
        info.p = [fx, 0.0, cx, 0.0,
                  0.0, fy, cy, 0.0,
                  0.0, 0.0, 1.0, 0.0]
        info.distortion_model = "plumb_bob"
        info.d = []  # no lens distortion on a virtual camera

        self.pub_info.publish(info)

        if self.debug_view:
            cv2.imshow("DA2 RGB", frame_bgr)
            dnorm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
            dcolor = cv2.applyColorMap(dnorm.astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imshow("DA2 Depth", dcolor)
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = DA2RGBD()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()
