import os, cv2, numpy as np, torch
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from datetime import timedelta

def letterbox_bgr_to_size(frame_bgr, H, W):
    """Resize with aspect ratio + center pad to (H,W). Returns rgb_small (H,W,3) and pad (top,left,nh,nw)."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    s = min(W / float(w), H / float(h))
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    top = (H - nh) // 2
    left = (W - nw) // 2
    canvas[top:top+nh, left:left+nw] = resized
    return canvas, (top, left, nh, nw)

class ZoeRGBD(Node):
    def __init__(self):
        super().__init__('zoe_rgbd')
        # --- params ---
        self.declare_parameter('video_source', '/home/fliight/mp4_rgbd_ws/video.mp4')
        self.declare_parameter('frame_id', 'camera_color_optical_frame')
        self.declare_parameter('img_height', 192)   # multiples of 32 recommended
        self.declare_parameter('img_width', 320)
        self.declare_parameter('use_gpu', True)
        self.declare_parameter('use_fp16', True)
        self.declare_parameter('show_debug', False)

        video_path = self.get_parameter('video_source').value
        self.frame_id = self.get_parameter('frame_id').value
        H = int(self.get_parameter('img_height').value)
        W = int(self.get_parameter('img_width').value)
        self.size = (H, W)

        # Open video with FFMPEG (works for your file)
        self.get_logger().info(f"Opening video file: {video_path}")
        self.cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {video_path}")
        try:
            backend_name = self.cap.getBackendName()
        except Exception:
            backend_name = "UNKNOWN"
        self.get_logger().info(f"VideoCapture opened: backend={backend_name}")

        # ZoeDepth init (inference config)
        conf = get_config("zoedepth", "infer", pretrained=True)
        self.device = "cuda" if (self.get_parameter('use_gpu').value and torch.cuda.is_available()) else "cpu"
        self.model = build_model(conf).to(self.device).eval()
        torch.set_grad_enabled(False)
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True

        # Publishers
        self.bridge = CvBridge()
        self.pub_rgb   = self.create_publisher(Image, '/camera/image_raw', 10)
        self.pub_depth = self.create_publisher(Image, '/camera/depth/image', 10)
        self.pub_info  = self.create_publisher(CameraInfo, '/camera/camera_info', 10)

        # Derive timer period from file FPS (fallback 10 Hz)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        target_hz = float(fps) if fps and fps > 0 else 10.0
        period = max(1.0/target_hz, 0.01)
        self.timer = self.create_timer(period, self.timer_callback)

        self.get_logger().info(f"ZoeDepth RGB-D publisher started. device={self.device} "
                               f"img={W}x{H} period={period:.3f}s (~{1.0/period:.1f} Hz)")

        self.show_debug = bool(self.get_parameter('show_debug').value)

    def timer_callback(self):
        ok, frame = self.cap.read()
        if not ok:
            self.get_logger().info("End of video, restarting…")
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return

        H, W = self.size
        rgb_small, pad = letterbox_bgr_to_size(frame, H, W)  # (H,W,3), uint8
        # -> tensor [1,3,H,W] on device
        img = torch.from_numpy(rgb_small).to(self.device).float() / 255.0
        img = img.permute(2, 0, 1).unsqueeze(0).contiguous()

        # inference
        if self.device == "cuda" and self.get_parameter('use_fp16').value:
            with torch.inference_mode(), torch.cuda.amp.autocast():
                d = self.model.infer(img, pad_input=False)[0, 0].float().cpu().numpy()
        else:
            with torch.inference_mode():
                d = self.model.infer(img, pad_input=False)[0, 0].cpu().numpy()

        depth = d.astype(np.float32)  # (H,W)

        # Publish (small, letterboxed) RGB + Depth + CameraInfo
        stamp = self.get_clock().now().to_msg()

        msg_rgb = self.bridge.cv2_to_imgmsg(rgb_small, encoding='rgb8')
        msg_rgb.header.frame_id = self.frame_id
        msg_rgb.header.stamp = stamp
        self.pub_rgb.publish(msg_rgb)

        msg_depth = self.bridge.cv2_to_imgmsg(depth, encoding='32FC1')
        msg_depth.header.frame_id = self.frame_id
        msg_depth.header.stamp = stamp
        self.pub_depth.publish(msg_depth)

        info = CameraInfo()
        info.header.stamp = stamp
        info.header.frame_id = self.frame_id
        info.width, info.height = W, H

        # Intrinsics guess scaled to output size (better than constants)
        # Base guess: fx=fy=525 at width=640 -> scale with width
        fx = fy = 525.0 * (W / 640.0)
        cx, cy = W / 2.0, H / 2.0
        info.k = [fx, 0.0, cx,
                  0.0, fy, cy,
                  0.0, 0.0, 1.0]
        # Identity distortion (unknown)
        info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        info.r = [1.0, 0.0, 0.0,
                  0.0, 1.0, 0.0,
                  0.0, 0.0, 1.0]
        info.p = [fx, 0.0, cx, 0.0,
                  0.0, fy, cy, 0.0,
                  0.0, 0.0, 1.0, 0.0]
        self.pub_info.publish(info)

        if self.show_debug:
            cv2.imshow("RGB small", cv2.cvtColor(rgb_small, cv2.COLOR_RGB2BGR))
            d8 = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            cv2.imshow("Depth small", cv2.applyColorMap(d8, cv2.COLORMAP_MAGMA))
            cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = ZoeRGBD()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
