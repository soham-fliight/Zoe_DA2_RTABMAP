
import os
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
import torch
import torchvision.transforms as T

torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))

@torch.inference_mode()
def load_zoe():
    # CPU-friendly ZoeD_NK
    model = torch.hub.load('isl-org/ZoeDepth', 'ZoeD_NK', pretrained=True)
    model.eval()
    return model

def np_to_image(rgb: np.ndarray, frame_id: str, stamp) -> Image:
    # rgb: HxWx3 uint8 in RGB order
    msg = Image()
    msg.header = Header(stamp=stamp, frame_id=frame_id)
    msg.height, msg.width = rgb.shape[:2]
    msg.encoding = 'rgb8'
    msg.is_bigendian = 0
    msg.step = msg.width * 3
    msg.data = rgb.tobytes()
    return msg

def depth_to_image(depth_m: np.ndarray, frame_id: str, stamp) -> Image:
    # depth_m: HxW float32 meters
    msg = Image()
    msg.header = Header(stamp=stamp, frame_id=frame_id)
    msg.height, msg.width = depth_m.shape
    msg.encoding = '32FC1'
    msg.is_bigendian = 0
    msg.step = msg.width * 4
    msg.data = depth_m.tobytes()
    return msg

class ZoeRGBD(Node):
    def __init__(self):
        super().__init__('zoe_rgbd')
        # params
        self.declare_parameter('video_source', '')
        self.declare_parameter('img_size_w', 640)
        self.declare_parameter('img_size_h', 480)
        self.declare_parameter('frame_id', 'camera_color_optical_frame')
        self.declare_parameter('fx', 525.0); self.declare_parameter('fy', 525.0)
        self.declare_parameter('cx', 320.0); self.declare_parameter('cy', 240.0)
        self.declare_parameter('fps', 30.0)
        self.declare_parameter('depth_scale', 1.0)

        self.w = int(self.get_parameter('img_size_w').value)
        self.h = int(self.get_parameter('img_size_h').value)
        self.fx = float(self.get_parameter('fx').value)
        self.fy = float(self.get_parameter('fy').value)
        self.cx = float(self.get_parameter('cx').value)
        self.cy = float(self.get_parameter('cy').value)
        self.fps = float(self.get_parameter('fps').value)
        self.frame_id = self.get_parameter('frame_id').value
        self.depth_scale = float(self.get_parameter('depth_scale').value)

        # ---- Source resolution & diagnostics ----
        src = self.get_parameter('video_source').value
        # allow hardcoded fallback via env or default path
        if not src:
            src = os.environ.get('ZOE_VIDEO', '/home/fliight/mp4_rgbd_ws/video.mp4')

        # if src looks like an int, treat as camera index
        cam_index = None
        if isinstance(src, str) and src.strip().isdigit():
            cam_index = int(src.strip())

        if cam_index is not None:
            self.get_logger().info(f"Opening camera index {cam_index}")
        else:
            if not os.path.exists(src):
                self.get_logger().error(f"Video path does not exist: {src}")
            else:
                try:
                    size = os.path.getsize(src)
                    self.get_logger().info(f"Opening video file: {src} ({size} bytes)")
                except Exception as e:
                    self.get_logger().warn(f"Could not stat file: {e}")

        # Try multiple backends
        self.cap = None
        candidates = []
        if cam_index is not None:
            candidates = [(cam_index, 0), (cam_index, cv2.CAP_FFMPEG), (cam_index, cv2.CAP_GSTREAMER)]
        else:
            candidates = [(src, cv2.CAP_FFMPEG), (src, cv2.CAP_GSTREAMER), (src, 0)]

        for source, backend in candidates:
            try:
                cap = cv2.VideoCapture(source, backend) if backend != 0 else cv2.VideoCapture(source)
                if cap.isOpened():
                    self.cap = cap
                    self.get_logger().info(f"VideoCapture opened with backend={backend} source={source}")
                    break
                else:
                    self.get_logger().warn(f"Open failed with backend={backend} source={source}")
            except Exception as e:
                self.get_logger().warn(f"Exception opening backend={backend} source={source}: {e}")

        if self.cap is None or not self.cap.isOpened():
            info = cv2.getBuildInformation()
            ffmpeg_supported = ("FFMPEG:" in info) or ("FFmpeg" in info)
            gst_supported = ("GStreamer:" in info) or ("GStreamer" in info)
            self.get_logger().error(
                "Cannot open video source. Diagnostics:\n"
                f"  src='{src}' cam_index={cam_index}\n"
                f"  OpenCV FFMPEG support: {ffmpeg_supported}\n"
                f"  OpenCV GStreamer support: {gst_supported}\n"
                "Install codecs if needed:\n"
                "  sudo apt install -y ffmpeg gstreamer1.0-libav\n"
            )
            raise RuntimeError(f"Cannot open video source: {src}")

        # Some sources don't report FPS; set desired FPS for timer regardless
        # Optionally, we can try to set frame size (for webcams)
        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)
        except Exception:
            pass

        self.model = load_zoe()
        self.to_tensor = T.Compose([T.ToTensor()])

        self.pub_rgb   = self.create_publisher(Image, '/camera/image_raw', 10)
        self.pub_depth = self.create_publisher(Image, '/camera/depth/image', 10)
        self.pub_info  = self.create_publisher(CameraInfo, '/camera/camera_info', 10)

        self.timer = self.create_timer(1.0 / max(1.0, self.fps), self.step)
        self.get_logger().info("ZoeDepth RGB-D publisher started.")

    def publish_info(self, stamp):
        ci = CameraInfo()
        ci.header.stamp = stamp
        ci.header.frame_id = self.frame_id
        ci.height = self.h; ci.width = self.w
        ci.k = [self.fx, 0.0, self.cx, 0.0, self.fy, self.cy, 0.0, 0.0, 1.0]
        ci.p = [self.fx, 0.0, self.cx, 0.0, 0.0, self.fy, self.cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        self.pub_info.publish(ci)

    def step(self):
        ok, bgr = self.cap.read()
        if not ok or bgr is None:
            # try to loop file sources
            self.get_logger().warn("Frame grab failed; attempting to loop/reopen...")
            try:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, bgr = self.cap.read()
            except Exception:
                ok = False
            if not ok or bgr is None:
                self.get_logger().error("Still cannot read frames; skipping this tick.")
                return

        # Resize to requested output size if needed
        if bgr.shape[1] != self.w or bgr.shape[0] != self.h:
            bgr = cv2.resize(bgr, (self.w, self.h), interpolation=cv2.INTER_AREA)

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # depth inference (meters)
        inp = self.to_tensor(rgb).unsqueeze(0)
        with torch.inference_mode():
            depth = self.model.infer(inp)[0, 0].cpu().numpy().astype(np.float32)

        depth *= self.depth_scale

        stamp = self.get_clock().now().to_msg()
        self.pub_rgb.publish(np_to_image(rgb, self.frame_id, stamp))
        self.pub_depth.publish(depth_to_image(depth, self.frame_id, stamp))
        self.publish_info(stamp)

def main():
    rclpy.init()
    node = ZoeRGBD()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
