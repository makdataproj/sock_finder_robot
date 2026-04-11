#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from collections import OrderedDict, deque, namedtuple
from pathlib import Path
from statistics import median

import cv2
import numpy as np
import torch
import tensorrt as trt

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Bool, Float32MultiArray

from ultralytics.utils.ops import scale_boxes

SOCK_ENGINE_PATH = "/home/jetson/models/seg_best_fp16.engine"
PERSON_ENGINE_PATH = "/home/jetson/models/legs_person_b1_fp16.engine"
IMGSZ = 640
DEVICE = "cuda:0"


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), scaleup=True, stride=32):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2.0
    dh /= 2.0

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


def preprocess_bgr(frame_bgr: np.ndarray, imgsz: int = 640, device: str = "cuda:0"):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img, ratio, dwdh = letterbox(frame_rgb, (imgsz, imgsz), stride=32)

    img = img.transpose((2, 0, 1))
    img = np.ascontiguousarray(img, dtype=np.float32)
    img /= 255.0

    tensor = torch.from_numpy(img).to(device)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)

    return tensor, ratio, dwdh


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def pick_pred_and_proto(outputs):
    proto = None
    pred = None
    for out in outputs:
        shape = tuple(out.shape)
        if len(shape) == 4:
            proto = out
        elif len(shape) == 3:
            pred = out
    if pred is None or proto is None:
        raise RuntimeError(f"Could not identify pred/proto tensors. Shapes: {[tuple(x.shape) for x in outputs]}")
    return pred, proto


def pick_pred_only(outputs):
    for out in outputs:
        if len(tuple(out.shape)) == 3:
            return out
    raise RuntimeError(f"Could not identify pred tensor. Shapes: {[tuple(x.shape) for x in outputs]}")


def decode_mask_single(proto_tensor: torch.Tensor, coeff_tensor: torch.Tensor) -> np.ndarray:
    proto = proto_tensor.detach().float().cpu().numpy()
    coeff = coeff_tensor.detach().float().cpu().numpy()
    nm, mh, mw = proto.shape
    proto_flat = proto.reshape(nm, -1)
    mask_flat = coeff @ proto_flat
    mask = sigmoid_np(mask_flat).reshape(mh, mw)
    return mask


def unletterbox_mask(mask_small: np.ndarray, orig_shape, imgsz: int, dwdh):
    orig_h, orig_w = orig_shape[:2]
    dw, dh = dwdh

    mask_net = cv2.resize(mask_small, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)

    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))

    y1 = top
    y2 = imgsz - bottom
    x1 = left
    x2 = imgsz - right

    if y2 <= y1 or x2 <= x1:
        return np.zeros((orig_h, orig_w), dtype=np.float32)

    mask_unpadded = mask_net[y1:y2, x1:x2]
    mask_orig = cv2.resize(mask_unpadded, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return mask_orig


def bbox_center(x1, y1, x2, y2):
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def bbox_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


class TRTModel:
    def __init__(self, engine_path: str, device: torch.device):
        self.device = device
        self.engine_path = str(Path(engine_path).expanduser())
        if not Path(self.engine_path).exists():
            raise FileNotFoundError(f"TensorRT engine not found: {self.engine_path}")

        Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(self.trt_logger, namespace="")

        with open(self.engine_path, "rb") as f, trt.Runtime(self.trt_logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {self.engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"Failed to create TensorRT execution context: {self.engine_path}")

        self.bindings = OrderedDict()
        self.input_name = None
        self.output_names = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_names.append(name)

        if self.input_name is None:
            raise RuntimeError(f"Could not find TRT input tensor for {self.engine_path}")
        if not self.output_names:
            raise RuntimeError(f"Could not find TRT output tensors for {self.engine_path}")

        input_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        if any(dim == -1 for dim in input_shape):
            self.context.set_input_shape(self.input_name, (1, 3, IMGSZ, IMGSZ))

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = tuple(self.context.get_tensor_shape(name))
            shape = tuple(1 if d < 0 else d for d in shape)
            torch_dtype = torch.from_numpy(np.empty((), dtype=dtype)).dtype
            data = torch.empty(size=shape, dtype=torch_dtype, device=self.device)
            self.bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))

        for name, binding in self.bindings.items():
            self.context.set_tensor_address(name, binding.ptr)

        dummy = torch.zeros((1, 3, IMGSZ, IMGSZ), dtype=torch.float32, device=self.device)
        self.predict(dummy)

    def predict(self, img_tensor: torch.Tensor):
        expected_shape = tuple(self.bindings[self.input_name].data.shape)
        if tuple(img_tensor.shape) != expected_shape:
            raise RuntimeError(f"Input shape mismatch. Expected {expected_shape}, got {tuple(img_tensor.shape)}")
        self.bindings[self.input_name].data.copy_(img_tensor)
        stream = torch.cuda.current_stream().cuda_stream
        ok = self.context.execute_async_v3(stream_handle=stream)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed")
        torch.cuda.synchronize()
        return [self.bindings[name].data for name in self.output_names]


class CombinedSockPersonTRTNode(Node):
    def __init__(self):
        super().__init__("combined_sock_person_trt_node")

        self.bridge = CvBridge()
        self.device = torch.device(DEVICE)

        self.declare_parameter("image_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/depth/image_raw")
        self.declare_parameter("sock_enable_topic", "/sock/detector_enable")
        self.declare_parameter("person_enable_topic", "/person/detector_enable")
        self.declare_parameter("mask_target_topic", "/sock/mask_target")
        self.declare_parameter("bbox_topic", "/sock/best_bbox")
        self.declare_parameter("person_target_topic", "/person/target")
        self.declare_parameter("debug_image_topic", "detect_image")

        self.declare_parameter("depth_is_meters", False)
        self.declare_parameter("depth_patch_radius", 6)
        self.declare_parameter("min_depth_m", 0.08)
        self.declare_parameter("max_depth_m", 3.0)

        self.declare_parameter("history_len", 5)
        self.declare_parameter("max_jump_px", 90.0)

        self.declare_parameter("mask_threshold", 0.50)
        self.declare_parameter("sock_conf_threshold", 0.10)
        self.declare_parameter("sock_lock_conf_threshold", 0.90)
        self.declare_parameter("min_mask_pixels", 50)
        self.declare_parameter("stop_y_percentile", 97.0)

        self.declare_parameter("morph_kernel_size", 5)
        self.declare_parameter("grasp_alpha_x", 0.65)
        self.declare_parameter("grasp_alpha_y", 0.50)
        self.declare_parameter("grasp_band_top_ratio", 0.45)
        self.declare_parameter("grasp_band_bottom_ratio", 0.75)
        self.declare_parameter("grasp_min_row_width_px", 12)

        self.declare_parameter("sock_lock_timeout_sec", 0.75)
        self.declare_parameter("sock_hold_last_target", True)
        self.declare_parameter("sock_match_center_dist_px", 140.0)
        self.declare_parameter("sock_min_iou_for_match", 0.05)
        self.declare_parameter("sock_switch_conf_margin", 0.20)
        self.declare_parameter("sock_prefer_locked_target", True)

        self.declare_parameter("person_conf_threshold", 0.25)
        self.declare_parameter("person_lock_conf_threshold", 0.40)
        self.declare_parameter("person_hold_last_target", True)
        self.declare_parameter("person_lock_timeout_sec", 0.75)
        self.declare_parameter("person_match_center_dist_px", 180.0)
        self.declare_parameter("person_min_iou_for_match", 0.05)
        self.declare_parameter("person_switch_conf_margin", 0.15)
        self.declare_parameter("person_class_id", 1)

        gp = lambda name: self.get_parameter(name).value
        self.image_topic = str(gp("image_topic"))
        self.depth_topic = str(gp("depth_topic"))
        self.sock_enable_topic = str(gp("sock_enable_topic"))
        self.person_enable_topic = str(gp("person_enable_topic"))
        self.mask_target_topic = str(gp("mask_target_topic"))
        self.bbox_topic = str(gp("bbox_topic"))
        self.person_target_topic = str(gp("person_target_topic"))
        self.debug_image_topic = str(gp("debug_image_topic"))

        self.depth_is_meters = bool(gp("depth_is_meters"))
        self.depth_patch_radius = int(gp("depth_patch_radius"))
        self.min_depth_m = float(gp("min_depth_m"))
        self.max_depth_m = float(gp("max_depth_m"))

        self.history_len = int(gp("history_len"))
        self.max_jump_px = float(gp("max_jump_px"))

        self.mask_threshold = float(gp("mask_threshold"))
        self.sock_conf_threshold = float(gp("sock_conf_threshold"))
        self.sock_lock_conf_threshold = float(gp("sock_lock_conf_threshold"))
        self.min_mask_pixels = int(gp("min_mask_pixels"))
        self.stop_y_percentile = float(gp("stop_y_percentile"))

        self.morph_kernel_size = int(gp("morph_kernel_size"))
        self.grasp_alpha_x = float(gp("grasp_alpha_x"))
        self.grasp_alpha_y = float(gp("grasp_alpha_y"))
        self.grasp_band_top_ratio = float(gp("grasp_band_top_ratio"))
        self.grasp_band_bottom_ratio = float(gp("grasp_band_bottom_ratio"))
        self.grasp_min_row_width_px = int(gp("grasp_min_row_width_px"))

        self.sock_lock_timeout_sec = float(gp("sock_lock_timeout_sec"))
        self.sock_hold_last_target = bool(gp("sock_hold_last_target"))
        self.sock_match_center_dist_px = float(gp("sock_match_center_dist_px"))
        self.sock_min_iou_for_match = float(gp("sock_min_iou_for_match"))
        self.sock_switch_conf_margin = float(gp("sock_switch_conf_margin"))
        self.sock_prefer_locked_target = bool(gp("sock_prefer_locked_target"))

        self.person_conf_threshold = float(gp("person_conf_threshold"))
        self.person_lock_conf_threshold = float(gp("person_lock_conf_threshold"))
        self.person_hold_last_target = bool(gp("person_hold_last_target"))
        self.person_lock_timeout_sec = float(gp("person_lock_timeout_sec"))
        self.person_match_center_dist_px = float(gp("person_match_center_dist_px"))
        self.person_min_iou_for_match = float(gp("person_min_iou_for_match"))
        self.person_switch_conf_margin = float(gp("person_switch_conf_margin"))
        self.person_class_id = int(gp("person_class_id"))

        self.latest_depth = None
        self.sock_enabled = True
        self.person_enabled = False

        self.sock_grasp_x_hist = deque(maxlen=self.history_len)
        self.sock_grasp_y_hist = deque(maxlen=self.history_len)
        self.sock_stop_y_hist = deque(maxlen=self.history_len)
        self.sock_conf_hist = deque(maxlen=self.history_len)
        self.sock_prev_grasp_x = None
        self.sock_prev_grasp_y = None

        self.sock_locked = False
        self.sock_locked_bbox = None
        self.sock_locked_grasp = None
        self.sock_locked_stop_y = None
        self.sock_locked_conf = 0.0
        self.sock_locked_depth_m = None
        self.sock_last_seen_time = 0.0

        self.person_locked = False
        self.person_locked_bbox = None
        self.person_locked_conf = 0.0
        self.person_last_seen_time = 0.0

        self.pub_sock_target = self.create_publisher(Float32MultiArray, self.mask_target_topic, 10)
        self.pub_sock_bbox = self.create_publisher(Float32MultiArray, self.bbox_topic, 10)
        self.pub_person_target = self.create_publisher(Float32MultiArray, self.person_target_topic, 10)
        self.pub_img = self.create_publisher(Image, self.debug_image_topic, 1)

        self.create_subscription(Image, self.image_topic, self.cb_image, 10)
        self.create_subscription(Image, self.depth_topic, self.cb_depth, 10)
        self.create_subscription(Bool, self.sock_enable_topic, self.cb_sock_enable, 10)
        self.create_subscription(Bool, self.person_enable_topic, self.cb_person_enable, 10)

        self.sock_model = TRTModel(SOCK_ENGINE_PATH, self.device)
        self.person_model = TRTModel(PERSON_ENGINE_PATH, self.device)

        self.last_shape_log_time = 0.0
        self.last_det_log_time = 0.0
        self.last_lock_log_time = 0.0

        self.get_logger().info(f"Loaded sock TRT engine: {SOCK_ENGINE_PATH}")
        self.get_logger().info(f"Loaded person TRT engine: {PERSON_ENGINE_PATH}")
        self.get_logger().info(f"Publishing /sock/mask_target on: {self.mask_target_topic}")
        self.get_logger().info(f"Publishing /person/target on: {self.person_target_topic}")
        self.get_logger().info(f"Listening for sock enable on: {self.sock_enable_topic}")
        self.get_logger().info(f"Listening for person enable on: {self.person_enable_topic}")

    def cb_sock_enable(self, msg: Bool):
        new_enabled = bool(msg.data)
        if self.sock_enabled == new_enabled:
            return
        self.sock_enabled = new_enabled
        if not self.sock_enabled:
            self.get_logger().info("Sock detector disabled")
            self._clear_sock_lock()
        else:
            self.get_logger().info("Sock detector enabled")

    def cb_person_enable(self, msg: Bool):
        new_enabled = bool(msg.data)
        if self.person_enabled == new_enabled:
            return
        self.person_enabled = new_enabled
        if not self.person_enabled:
            self.get_logger().info("Person detector disabled")
            self._clear_person_lock()
        else:
            self.get_logger().info("Person detector enabled")

    def cb_depth(self, msg: Image):
        try:
            if self.depth_is_meters:
                self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
            else:
                self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1")
        except Exception as e:
            self.get_logger().error(f"Depth conversion failed: {e}")

    def _get_depth_median(self, u: int, v: int):
        if self.latest_depth is None:
            return None
        depth = self.latest_depth
        h, w = depth.shape[:2]
        u = int(np.clip(u, 0, w - 1))
        v = int(np.clip(v, 0, h - 1))
        r = max(1, self.depth_patch_radius)
        x1 = max(0, u - r)
        x2 = min(w, u + r + 1)
        y1 = max(0, v - r)
        y2 = min(h, v + r + 1)
        patch = depth[y1:y2, x1:x2]
        vals = patch[np.isfinite(patch)]
        if vals.size == 0:
            return None
        if self.depth_is_meters:
            vals = vals[(vals > self.min_depth_m) & (vals < self.max_depth_m)]
            if vals.size == 0:
                return None
            return float(np.median(vals))
        vals = vals[(vals > 1) & (vals < int(self.max_depth_m * 1000.0))]
        if vals.size == 0:
            return None
        return float(np.median(vals)) / 1000.0

    def _clear_sock_lock(self):
        self.sock_locked = False
        self.sock_locked_bbox = None
        self.sock_locked_grasp = None
        self.sock_locked_stop_y = None
        self.sock_locked_conf = 0.0
        self.sock_locked_depth_m = None
        self.sock_last_seen_time = 0.0
        self.sock_grasp_x_hist.clear()
        self.sock_grasp_y_hist.clear()
        self.sock_stop_y_hist.clear()
        self.sock_conf_hist.clear()
        self.sock_prev_grasp_x = None
        self.sock_prev_grasp_y = None

    def _clear_person_lock(self):
        self.person_locked = False
        self.person_locked_bbox = None
        self.person_locked_conf = 0.0
        self.person_last_seen_time = 0.0

    def _publish_sock_target(self, grasp_x, grasp_y, stop_y, conf, bbox):
        x1, y1, x2, y2 = bbox
        target_msg = Float32MultiArray()
        target_msg.data = [float(grasp_x), float(grasp_y), float(stop_y), float(conf)]
        self.pub_sock_target.publish(target_msg)
        bbox_msg = Float32MultiArray()
        bbox_msg.data = [float(x1), float(y1), float(x2), float(y2), float(conf)]
        self.pub_sock_bbox.publish(bbox_msg)

    def _publish_person_target(self, cx, cy, width, conf):
        msg = Float32MultiArray()
        msg.data = [float(cx), float(cy), float(width), float(conf)]
        self.pub_person_target.publish(msg)

    def _publish_debug(self, src_msg: Image, image: np.ndarray):
        try:
            out_msg = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
            out_msg.header = src_msg.header
            self.pub_img.publish(out_msg)
        except Exception as e:
            self.get_logger().error(f"Debug publish failed: {e}")

    def _draw_sock_locked_state(self, annotated, status_text="SOCK"):
        if self.sock_locked_bbox is None or self.sock_locked_grasp is None:
            return annotated
        x1, y1, x2, y2 = [int(v) for v in self.sock_locked_bbox]
        gx, gy = [int(v) for v in self.sock_locked_grasp]
        sy = int(self.sock_locked_stop_y) if self.sock_locked_stop_y is not None else y2
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 200, 0), 2)
        cv2.circle(annotated, (gx, gy), 8, (0, 255, 255), -1)
        cv2.line(annotated, (x1, sy), (x2, sy), (0, 0, 255), 2)
        cv2.putText(annotated, f"{status_text} conf={self.sock_locked_conf:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2, cv2.LINE_AA)
        return annotated

    def _draw_person_locked_state(self, annotated, status_text="PERSON"):
        if self.person_locked_bbox is None:
            return annotated
        x1, y1, x2, y2 = [int(v) for v in self.person_locked_bbox]
        cx = int(0.5 * (x1 + x2))
        cy = int(0.5 * (y1 + y2))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(annotated, (cx, cy), 6, (0, 255, 255), -1)
        cv2.putText(annotated, f"{status_text} conf={self.person_locked_conf:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
        return annotated

    def _sock_smooth(self, grasp_x, grasp_y, stop_y, conf, allow_large_jump=False):
        if (not allow_large_jump and self.sock_prev_grasp_x is not None and self.sock_prev_grasp_y is not None):
            jump = np.hypot(grasp_x - self.sock_prev_grasp_x, grasp_y - self.sock_prev_grasp_y)
            if jump > self.max_jump_px:
                return None
        self.sock_grasp_x_hist.append(float(grasp_x))
        self.sock_grasp_y_hist.append(float(grasp_y))
        self.sock_stop_y_hist.append(float(stop_y))
        self.sock_conf_hist.append(float(conf))
        gx = median(self.sock_grasp_x_hist)
        gy = median(self.sock_grasp_y_hist)
        sy = median(self.sock_stop_y_hist)
        cf = median(self.sock_conf_hist)
        self.sock_prev_grasp_x = gx
        self.sock_prev_grasp_y = gy
        return gx, gy, sy, cf

    def _select_sock_grasp_from_thickest_band(self, mask_clean: np.ndarray, xs: np.ndarray, ys: np.ndarray):
        y_min = int(np.min(ys))
        y_max = int(np.max(ys))
        sock_h = max(1, y_max - y_min + 1)
        band_top = int(round(y_min + self.grasp_band_top_ratio * sock_h))
        band_bottom = int(round(y_min + self.grasp_band_bottom_ratio * sock_h))
        band_top = max(y_min, min(y_max, band_top))
        band_bottom = max(band_top, min(y_max, band_bottom))
        best = None
        for row_y in range(band_top, band_bottom + 1):
            row_xs = np.where(mask_clean[row_y] > 0)[0]
            if row_xs.size == 0:
                continue
            row_left = int(row_xs[0])
            row_right = int(row_xs[-1])
            row_width = row_right - row_left + 1
            if row_width < self.grasp_min_row_width_px:
                continue
            row_center_x = 0.5 * (row_left + row_right)
            edge_margin = min(row_center_x - row_left, row_right - row_center_x)
            score = (row_width, edge_margin)
            if best is None or score > best[0]:
                best = (score, row_center_x, float(row_y), row_width, row_left, row_right, band_top, band_bottom)
        if best is None:
            return None
        _, row_center_x, row_y, row_width, row_left, row_right, band_top, band_bottom = best
        return {
            "x": float(row_center_x), "y": float(row_y), "row_width": int(row_width),
            "row_left": int(row_left), "row_right": int(row_right),
            "band_top": int(band_top), "band_bottom": int(band_bottom),
        }

    def _extract_sock_candidate(self, det_row, proto_single, frame_shape, dwdh):
        x1, y1, x2, y2 = map(int, det_row[:4].tolist())
        conf = float(det_row[4].item())
        coeff = det_row[6:38]
        mask_small = decode_mask_single(proto_single, coeff)
        mask_orig = unletterbox_mask(mask_small, frame_shape, IMGSZ, dwdh)
        mask_bin = (mask_orig > self.mask_threshold).astype(np.uint8)

        bbox_mask = np.zeros_like(mask_bin, dtype=np.uint8)
        x1c = max(0, min(mask_bin.shape[1] - 1, x1))
        x2c = max(0, min(mask_bin.shape[1], x2))
        y1c = max(0, min(mask_bin.shape[0] - 1, y1))
        y2c = max(0, min(mask_bin.shape[0], y2))
        bbox_mask[y1c:y2c, x1c:x2c] = 1
        mask_bin = (mask_bin * bbox_mask).astype(np.uint8)

        k = max(3, self.morph_kernel_size)
        if k % 2 == 0:
            k += 1
        kernel = np.ones((k, k), np.uint8)
        mask_clean = cv2.morphologyEx(mask_bin.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_clean, connectivity=8)
        if num_labels > 1:
            largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask_clean = (labels == largest_idx).astype(np.uint8)
        else:
            mask_clean = mask_clean.astype(np.uint8)

        ys, xs = np.where(mask_clean > 0)
        if len(xs) < self.min_mask_pixels:
            grasp_x = int((x1 + x2) * 0.5)
            grasp_y = int((y1 + y2) * 0.5)
            stop_y = int(y2)
            mask_for_overlay = mask_clean
            row_width_px = 0
            thickest_row_y = grasp_y
            thickest_left = x1
            thickest_right = x2
            band_top = y1
            band_bottom = y2

            seg_width_px = max(0, x2 - x1)
            seg_height_px = max(0, y2 - y1)
            sock_orientation = "in grip" 
            if seg_width_px > seg_height_px:
                sock_orientation = "vertical"
            elif seg_width_px < seg_height_px:
                sock_orientation = "horizontal"

            
            self.get_logger().info(
                f"SOCK_ORIENTATION_CHECK_FALLBACK: "
                f"seg_width_px={seg_width_px} seg_height_px={seg_height_px} "
                f"orientation={sock_orientation}"
            )
        else:
            seg_x_min = int(np.min(xs))
            seg_x_max = int(np.max(xs))
            seg_y_min = int(np.min(ys))
            seg_y_max = int(np.max(ys))

            seg_width_px = seg_x_max - seg_x_min + 1
            seg_height_px = seg_y_max - seg_y_min + 1
            sock_orientation = "horizontal" if seg_width_px > seg_height_px else "vertical"

            self.get_logger().info(
                f"SOCK_ORIENTATION_CHECK: "
                f"seg_width_px={seg_width_px} seg_height_px={seg_height_px} "
                f"orientation={sock_orientation}"
            )

            cx = float(np.mean(xs))
            cy = float(np.mean(ys))
            dist = cv2.distanceTransform(mask_clean, cv2.DIST_L2, 5)
            _, _, _, max_loc = cv2.minMaxLoc(dist)
            interior_x = float(max_loc[0])
            interior_y = float(max_loc[1])
            thickest = self._select_sock_grasp_from_thickest_band(mask_clean, xs, ys)
            if thickest is not None:
                thick_x = float(thickest["x"])
                thick_y = float(thickest["y"])
                row_width_px = int(thickest["row_width"])
                thickest_row_y = int(thickest["y"])
                thickest_left = int(thickest["row_left"])
                thickest_right = int(thickest["row_right"])
                band_top = int(thickest["band_top"])
                band_bottom = int(thickest["band_bottom"])
                grasp_x = int(round(self.grasp_alpha_x * interior_x + (1.0 - self.grasp_alpha_x) * thick_x))
                grasp_y = int(round(self.grasp_alpha_y * interior_y + (1.0 - self.grasp_alpha_y) * thick_y))
            else:
                row_width_px = 0
                thickest_row_y = int(round(cy))
                thickest_left = int(np.min(xs))
                thickest_right = int(np.max(xs))
                band_top = int(np.min(ys))
                band_bottom = int(np.max(ys))
                grasp_x = int(round(self.grasp_alpha_x * interior_x + (1.0 - self.grasp_alpha_x) * cx))
                grasp_y = int(round(self.grasp_alpha_y * interior_y + (1.0 - self.grasp_alpha_y) * cy))
            if (grasp_y < 0 or grasp_y >= mask_clean.shape[0] or grasp_x < 0 or grasp_x >= mask_clean.shape[1] or mask_clean[grasp_y, grasp_x] == 0):
                pts = np.column_stack((xs, ys)).astype(np.float32)
                target = np.array([[grasp_x, grasp_y]], dtype=np.float32)
                dists = np.sum((pts - target) ** 2, axis=1)
                nearest = pts[np.argmin(dists)]
                grasp_x = int(nearest[0])
                grasp_y = int(nearest[1])
            stop_y = int(np.percentile(ys, self.stop_y_percentile))
            mask_for_overlay = mask_clean
        depth_m = self._get_depth_median(grasp_x, grasp_y)
        return {
            "bbox": (x1, y1, x2, y2), "conf": conf, "grasp_x": grasp_x, "grasp_y": grasp_y,
            "stop_y": stop_y, "depth_m": depth_m, "mask_bin": mask_for_overlay,
            "row_width_px": row_width_px, "thickest_row_y": thickest_row_y,
            "thickest_left": thickest_left, "thickest_right": thickest_right,
            "band_top": band_top, "band_bottom": band_bottom,
            "seg_width_px": int(seg_width_px), "seg_height_px": int(seg_height_px),
            "sock_orientation": sock_orientation,
        }

    def _select_sock_candidate(self, candidates):
        if not candidates:
            return None
        if not self.sock_locked or self.sock_locked_bbox is None:
            return max(candidates, key=lambda c: c["conf"])
        locked_bbox = self.sock_locked_bbox
        locked_cx, locked_cy = bbox_center(*locked_bbox)
        matched, unmatched = [], []
        for c in candidates:
            cx, cy = bbox_center(*c["bbox"])
            dist = float(np.hypot(cx - locked_cx, cy - locked_cy))
            iou = bbox_iou(c["bbox"], locked_bbox)
            c["match_dist"] = dist
            c["match_iou"] = iou
            if dist <= self.sock_match_center_dist_px or iou >= self.sock_min_iou_for_match:
                matched.append(c)
            else:
                unmatched.append(c)
        if matched:
            matched.sort(key=lambda c: (c["match_dist"], -c["conf"]))
            return matched[0]
        best_unmatched = max(unmatched, key=lambda c: c["conf"]) if unmatched else None
        if best_unmatched is None:
            return None
        if not self.sock_prefer_locked_target:
            return best_unmatched
        if best_unmatched["conf"] >= (self.sock_locked_conf + self.sock_switch_conf_margin):
            return best_unmatched
        return None

    def _process_sock_frame(self, msg: Image, frame_bgr: np.ndarray, annotated: np.ndarray):
        now = time.time()
        img_tensor, _, dwdh = preprocess_bgr(frame_bgr, imgsz=IMGSZ, device=DEVICE)
        outputs = self.sock_model.predict(img_tensor)
        pred, proto = pick_pred_and_proto(outputs)

        if now - self.last_shape_log_time > 2.0:
            self.get_logger().info(f"sock pred shape={tuple(pred.shape)}, proto shape={tuple(proto.shape)}")
            self.last_shape_log_time = now

        if pred.ndim != 3:
            raise RuntimeError(f"Unexpected sock pred ndim: {pred.ndim}, shape={tuple(pred.shape)}")
        if pred.shape[1] == 38 and pred.shape[2] != 38:
            pred = pred.transpose(1, 2)
        pred = pred[0]
        if pred.ndim != 2 or pred.shape[1] != 38:
            raise RuntimeError(f"Unexpected sock pred shape after normalization: {tuple(pred.shape)}")

        confs = pred[:, 4]
        keep = confs > self.sock_conf_threshold
        det = pred[keep]
        if det.shape[0] > 0:
            det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], frame_bgr.shape).round()
            det = det[det[:, 5] == 0]

        candidates = []
        if det.shape[0] > 0:
            proto_single = proto[0]
            for i in range(det.shape[0]):
                c = self._extract_sock_candidate(det[i], proto_single, frame_bgr.shape, dwdh)
                if c["depth_m"] is not None:
                    candidates.append(c)

        selected = self._select_sock_candidate(candidates)
        if selected is not None and (not self.sock_locked) and selected["conf"] < self.sock_lock_conf_threshold:
            selected = None

        if selected is not None:
            smoothed = self._sock_smooth(selected["grasp_x"], selected["grasp_y"], selected["stop_y"], selected["conf"], allow_large_jump=not self.sock_locked)
            if smoothed is not None:
                gx, gy, sy, cf = smoothed
                self.sock_locked = True
                self.sock_locked_bbox = selected["bbox"]
                self.sock_locked_grasp = (gx, gy)
                self.sock_locked_stop_y = sy
                self.sock_locked_conf = cf
                self.sock_locked_depth_m = selected["depth_m"]
                self.sock_last_seen_time = now
                self.get_logger().info(
                    f"SOCK_LOCKED_ORIENTATION: orientation={selected['sock_orientation']} "
                    f"seg_width_px={selected['seg_width_px']} seg_height_px={selected['seg_height_px']}"
                )
                self._publish_sock_target(gx, gy, sy, cf, self.sock_locked_bbox)
                overlay = annotated.copy()
                overlay[selected["mask_bin"] > 0] = (0, 180, 255)
                annotated = cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0)
                band_top = int(selected["band_top"])
                band_bottom = int(selected["band_bottom"])
                thick_y = int(selected["thickest_row_y"])
                thick_l = int(selected["thickest_left"])
                thick_r = int(selected["thickest_right"])
                cv2.line(annotated, (0, band_top), (annotated.shape[1] - 1, band_top), (255, 0, 0), 1)
                cv2.line(annotated, (0, band_bottom), (annotated.shape[1] - 1, band_bottom), (255, 0, 0), 1)
                cv2.line(annotated, (thick_l, thick_y), (thick_r, thick_y), (0, 255, 0), 2)
                annotated = self._draw_sock_locked_state(annotated, status_text="SOCK")
                self._publish_debug(msg, annotated)
                return

        if self.sock_locked and self.sock_hold_last_target:
            age = now - self.sock_last_seen_time
            if age <= self.sock_lock_timeout_sec:
                gx, gy = self.sock_locked_grasp
                sy = self.sock_locked_stop_y
                cf = self.sock_locked_conf
                self._publish_sock_target(gx, gy, sy, cf, self.sock_locked_bbox)
                annotated = self._draw_sock_locked_state(annotated, status_text=f"SOCK HOLD {age:.2f}s")
                self._publish_debug(msg, annotated)
                return

        if self.sock_locked:
            self._clear_sock_lock()
        cv2.putText(annotated, "No sock locked", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        self._publish_debug(msg, annotated)

    def _pick_person_candidate(self, det):
        if det is None or len(det) == 0:
            return None
        people = []
        for row in det:
            cls_id = int(row[5].item())
            conf = float(row[4].item())
            if cls_id != self.person_class_id:
                continue
            x1, y1, x2, y2 = map(float, row[:4].tolist())
            people.append((x1, y1, x2, y2, conf))
        if not people:
            return None
        if not self.person_locked or self.person_locked_bbox is None:
            return max(people, key=lambda p: p[4])
        matched = []
        lcx, lcy = bbox_center(*self.person_locked_bbox)
        for p in people:
            box = p[:4]
            conf = p[4]
            iou = bbox_iou(box, self.person_locked_bbox)
            cx, cy = bbox_center(*box)
            dist = float(np.hypot(cx - lcx, cy - lcy))
            if dist <= self.person_match_center_dist_px or iou >= self.person_min_iou_for_match:
                matched.append((dist, -conf, p))
        if matched:
            matched.sort(key=lambda x: (x[0], x[1]))
            return matched[0][2]
        best = max(people, key=lambda p: p[4])
        if best[4] >= (self.person_locked_conf + self.person_switch_conf_margin):
            return best
        return None

    def _process_person_frame(self, msg: Image, frame_bgr: np.ndarray, annotated: np.ndarray):
        img_tensor, _, _ = preprocess_bgr(frame_bgr, imgsz=IMGSZ, device=DEVICE)
        outputs = self.person_model.predict(img_tensor)
        pred = pick_pred_only(outputs)
        if pred.shape[1] == 6 and pred.shape[2] != 6:
            pred = pred.transpose(1, 2)
        pred = pred[0]
        if pred.ndim != 2 or pred.shape[1] < 6:
            return
        confs = pred[:, 4]
        keep = confs > self.person_conf_threshold
        det = pred[keep]
        if det.shape[0] > 0:
            det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], frame_bgr.shape).round()
        selected = self._pick_person_candidate(det)
        now = time.time()
        if selected is None:
            if self.person_locked and self.person_hold_last_target and (now - self.person_last_seen_time) <= self.person_lock_timeout_sec:
                x1, y1, x2, y2 = self.person_locked_bbox
                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)
                width = x2 - x1
                self._publish_person_target(cx, cy, width, self.person_locked_conf)
                annotated = self._draw_person_locked_state(annotated, status_text=f"PERSON HOLD {now - self.person_last_seen_time:.2f}s")
                self._publish_debug(msg, annotated)
                return
            self._clear_person_lock()
            cv2.putText(annotated, "No person locked", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
            self._publish_debug(msg, annotated)
            return

        x1, y1, x2, y2, conf = selected
        if not self.person_locked and conf < self.person_lock_conf_threshold:
            cv2.putText(annotated, f"Person conf {conf:.2f} below lock", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
            self._publish_debug(msg, annotated)
            return
        self.person_locked = True
        self.person_locked_bbox = (x1, y1, x2, y2)
        self.person_locked_conf = conf
        self.person_last_seen_time = now
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        width = x2 - x1
        self._publish_person_target(cx, cy, width, conf)
        annotated = self._draw_person_locked_state(annotated, status_text="PERSON")
        self._publish_debug(msg, annotated)

    def cb_image(self, msg: Image):
        if not self.sock_enabled and not self.person_enabled:
            return
        try:
            frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            annotated = frame_bgr.copy()
            # Prefer person mode when both are accidentally enabled.
            if self.person_enabled:
                self._process_person_frame(msg, frame_bgr, annotated)
            elif self.sock_enabled:
                self._process_sock_frame(msg, frame_bgr, annotated)
        except Exception as e:
            self.get_logger().error(f"Inference callback failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = CombinedSockPersonTRTNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
