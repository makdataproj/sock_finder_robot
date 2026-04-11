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


TRT_ENGINE_PATH = "/home/jetson/models/seg_best_fp16.engine"
IMGSZ = 640
DEVICE = "cuda:0"

NAMES = ["sock"]


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
    """
    Expected TRT segmentation outputs:
      pred  -> [1, N, 38] or [1, 38, N]
      proto -> [1, 32, 160, 160]

    Where 38 = [x1, y1, x2, y2, conf, cls, 32 mask coeffs]
    """
    proto = None
    pred = None

    for out in outputs:
        shape = tuple(out.shape)
        if len(shape) == 4:
            proto = out
        elif len(shape) == 3:
            pred = out

    if pred is None or proto is None:
        raise RuntimeError(
            f"Could not identify pred/proto tensors. Shapes: {[tuple(x.shape) for x in outputs]}"
        )

    return pred, proto


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


class SockMaskTargetTRTNode(Node):
    def __init__(self):
        super().__init__("sock_mask_target_trt_node")

        self.bridge = CvBridge()
        self.device = torch.device(DEVICE)

        self.declare_parameter("image_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/depth/image_raw")
        self.declare_parameter("mask_target_topic", "/sock/mask_target")
        self.declare_parameter("bbox_topic", "/sock/best_bbox")
        self.declare_parameter("debug_image_topic", "detect_image")
        self.declare_parameter("enable_topic", "/sock/detector_enable")

        self.declare_parameter("depth_is_meters", False)
        self.declare_parameter("depth_patch_radius", 6)
        self.declare_parameter("min_depth_m", 0.08)
        self.declare_parameter("max_depth_m", 3.0)

        self.declare_parameter("history_len", 5)
        self.declare_parameter("max_jump_px", 90.0)

        self.declare_parameter("mask_threshold", 0.50)
        self.declare_parameter("conf_threshold", 0.10)
        self.declare_parameter("lock_conf_threshold", 0.90)
        self.declare_parameter("min_mask_pixels", 50)
        self.declare_parameter("stop_y_percentile", 97.0)

        # Grasp-point tuning for thickest-area-of-sock behavior
        self.declare_parameter("morph_kernel_size", 5)
        self.declare_parameter("grasp_alpha_x", 0.65)
        self.declare_parameter("grasp_alpha_y", 0.50)
        self.declare_parameter("grasp_band_top_ratio", 0.45)
        self.declare_parameter("grasp_band_bottom_ratio", 0.75)
        self.declare_parameter("grasp_min_row_width_px", 12)

        # Lock / tracker params
        self.declare_parameter("lock_timeout_sec", 0.75)
        self.declare_parameter("hold_last_target", True)
        self.declare_parameter("match_center_dist_px", 140.0)
        self.declare_parameter("min_iou_for_match", 0.05)
        self.declare_parameter("switch_conf_margin", 0.20)
        self.declare_parameter("prefer_locked_target", True)

        self.image_topic = str(self.get_parameter("image_topic").value)
        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.mask_target_topic = str(self.get_parameter("mask_target_topic").value)
        self.bbox_topic = str(self.get_parameter("bbox_topic").value)
        self.debug_image_topic = str(self.get_parameter("debug_image_topic").value)
        self.enable_topic = str(self.get_parameter("enable_topic").value)

        self.depth_is_meters = bool(self.get_parameter("depth_is_meters").value)
        self.depth_patch_radius = int(self.get_parameter("depth_patch_radius").value)
        self.min_depth_m = float(self.get_parameter("min_depth_m").value)
        self.max_depth_m = float(self.get_parameter("max_depth_m").value)

        self.history_len = int(self.get_parameter("history_len").value)
        self.max_jump_px = float(self.get_parameter("max_jump_px").value)

        self.mask_threshold = float(self.get_parameter("mask_threshold").value)
        self.conf_threshold = float(self.get_parameter("conf_threshold").value)
        self.lock_conf_threshold = float(self.get_parameter("lock_conf_threshold").value)
        self.min_mask_pixels = int(self.get_parameter("min_mask_pixels").value)
        self.stop_y_percentile = float(self.get_parameter("stop_y_percentile").value)

        self.morph_kernel_size = int(self.get_parameter("morph_kernel_size").value)
        self.grasp_alpha_x = float(self.get_parameter("grasp_alpha_x").value)
        self.grasp_alpha_y = float(self.get_parameter("grasp_alpha_y").value)
        self.grasp_band_top_ratio = float(self.get_parameter("grasp_band_top_ratio").value)
        self.grasp_band_bottom_ratio = float(self.get_parameter("grasp_band_bottom_ratio").value)
        self.grasp_min_row_width_px = int(self.get_parameter("grasp_min_row_width_px").value)

        self.lock_timeout_sec = float(self.get_parameter("lock_timeout_sec").value)
        self.hold_last_target = bool(self.get_parameter("hold_last_target").value)
        self.match_center_dist_px = float(self.get_parameter("match_center_dist_px").value)
        self.min_iou_for_match = float(self.get_parameter("min_iou_for_match").value)
        self.switch_conf_margin = float(self.get_parameter("switch_conf_margin").value)
        self.prefer_locked_target = bool(self.get_parameter("prefer_locked_target").value)

        self.latest_depth = None

        self.enabled = True

        self.grasp_x_hist = deque(maxlen=self.history_len)
        self.grasp_y_hist = deque(maxlen=self.history_len)
        self.stop_y_hist = deque(maxlen=self.history_len)
        self.conf_hist = deque(maxlen=self.history_len)

        self.prev_grasp_x = None
        self.prev_grasp_y = None

        # Lock state
        self.locked = False
        self.locked_bbox = None
        self.locked_grasp = None
        self.locked_stop_y = None
        self.locked_conf = 0.0
        self.locked_depth_m = None
        self.last_seen_time = 0.0
        self.last_publish_time = 0.0

        self.pub_target = self.create_publisher(Float32MultiArray, self.mask_target_topic, 10)
        self.pub_bbox = self.create_publisher(Float32MultiArray, self.bbox_topic, 10)
        self.pub_img = self.create_publisher(Image, self.debug_image_topic, 1)

        self.create_subscription(Image, self.image_topic, self.cb_image, 10)
        self.create_subscription(Image, self.depth_topic, self.cb_depth, 10)
        self.create_subscription(Bool, self.enable_topic, self.cb_enable, 10)

        self._init_trt(TRT_ENGINE_PATH)

        self.last_shape_log_time = 0.0
        self.last_det_log_time = 0.0
        self.last_lock_log_time = 0.0

        self.get_logger().info(f"Loaded TRT engine: {TRT_ENGINE_PATH}")
        self.get_logger().info(f"Publishing /sock/mask_target on: {self.mask_target_topic}")
        self.get_logger().info(f"Publishing /sock/best_bbox on: {self.bbox_topic}")
        self.get_logger().info(f"Publishing debug image on: {self.debug_image_topic}")
        self.get_logger().info(f"Listening for enable on: {self.enable_topic}")
        self.get_logger().info(
            f"Mask grasp tuning: alpha_x={self.grasp_alpha_x:.2f}, "
            f"alpha_y={self.grasp_alpha_y:.2f}, kernel={self.morph_kernel_size}, "
            f"band=({self.grasp_band_top_ratio:.2f},{self.grasp_band_bottom_ratio:.2f}), "
            f"min_row_width={self.grasp_min_row_width_px}"
        )
        self.get_logger().info(
            f"New lock/reacquire requires conf >= {self.lock_conf_threshold:.2f}"
        )

    def cb_enable(self, msg: Bool):
        new_enabled = bool(msg.data)
        if self.enabled == new_enabled:
            return

        self.enabled = new_enabled

        if not self.enabled:
            self.get_logger().info("Sock detector disabled")
            self._clear_lock()
        else:
            self.get_logger().info("Sock detector enabled")

    def cb_depth(self, msg: Image):
        try:
            if self.depth_is_meters:
                self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
            else:
                self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1")
        except Exception as e:
            self.get_logger().error(f"Depth conversion failed: {e}")

    def _init_trt(self, engine_path: str):
        engine_path = str(Path(engine_path).expanduser())
        if not Path(engine_path).exists():
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")

        Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))

        self.trt_logger = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(self.trt_logger, namespace="")

        with open(engine_path, "rb") as f, trt.Runtime(self.trt_logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine")

        self.trt_context = self.engine.create_execution_context()
        if self.trt_context is None:
            raise RuntimeError("Failed to create TensorRT execution context")

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
            raise RuntimeError("Could not find TRT input tensor")
        if not self.output_names:
            raise RuntimeError("Could not find TRT output tensors")

        input_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        if any(dim == -1 for dim in input_shape):
            self.trt_context.set_input_shape(self.input_name, (1, 3, IMGSZ, IMGSZ))

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = tuple(self.trt_context.get_tensor_shape(name))
            shape = tuple(1 if d < 0 else d for d in shape)

            torch_dtype = torch.from_numpy(np.empty((), dtype=dtype)).dtype
            data = torch.empty(size=shape, dtype=torch_dtype, device=self.device)
            self.bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))

        for name, binding in self.bindings.items():
            self.trt_context.set_tensor_address(name, binding.ptr)

        dummy = torch.zeros((1, 3, IMGSZ, IMGSZ), dtype=torch.float32, device=self.device)
        _ = self._predict(dummy)

    def _predict(self, img_tensor: torch.Tensor):
        expected_shape = tuple(self.bindings[self.input_name].data.shape)
        if tuple(img_tensor.shape) != expected_shape:
            raise RuntimeError(
                f"Input shape mismatch. Expected {expected_shape}, got {tuple(img_tensor.shape)}"
            )

        self.bindings[self.input_name].data.copy_(img_tensor)

        stream = torch.cuda.current_stream().cuda_stream
        ok = self.trt_context.execute_async_v3(stream_handle=stream)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed")

        torch.cuda.synchronize()
        return [self.bindings[name].data for name in self.output_names]

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

    def _smooth(self, grasp_x, grasp_y, stop_y, conf, allow_large_jump=False):
        if (
            not allow_large_jump
            and self.prev_grasp_x is not None
            and self.prev_grasp_y is not None
        ):
            jump = np.hypot(grasp_x - self.prev_grasp_x, grasp_y - self.prev_grasp_y)
            if jump > self.max_jump_px:
                return None

        self.grasp_x_hist.append(float(grasp_x))
        self.grasp_y_hist.append(float(grasp_y))
        self.stop_y_hist.append(float(stop_y))
        self.conf_hist.append(float(conf))

        gx = median(self.grasp_x_hist)
        gy = median(self.grasp_y_hist)
        sy = median(self.stop_y_hist)
        cf = median(self.conf_hist)

        self.prev_grasp_x = gx
        self.prev_grasp_y = gy
        return gx, gy, sy, cf

    def _reset_smoothers(self):
        self.grasp_x_hist.clear()
        self.grasp_y_hist.clear()
        self.stop_y_hist.clear()
        self.conf_hist.clear()
        self.prev_grasp_x = None
        self.prev_grasp_y = None

    def _clear_lock(self):
        self.locked = False
        self.locked_bbox = None
        self.locked_grasp = None
        self.locked_stop_y = None
        self.locked_conf = 0.0
        self.locked_depth_m = None
        self.last_seen_time = 0.0
        self._reset_smoothers()

    def _publish_target(self, grasp_x, grasp_y, stop_y, conf, bbox):
        x1, y1, x2, y2 = bbox

        target_msg = Float32MultiArray()
        target_msg.data = [float(grasp_x), float(grasp_y), float(stop_y), float(conf)]
        self.pub_target.publish(target_msg)

        bbox_msg = Float32MultiArray()
        bbox_msg.data = [float(x1), float(y1), float(x2), float(y2), float(conf)]
        self.pub_bbox.publish(bbox_msg)

        self.last_publish_time = time.time()

    def _publish_debug(self, src_msg: Image, image: np.ndarray):
        try:
            out_msg = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
            out_msg.header = src_msg.header
            self.pub_img.publish(out_msg)
        except Exception as e:
            self.get_logger().error(f"Debug publish failed: {e}")

    def _draw_locked_state(self, annotated, status_text="LOCKED"):
        if self.locked_bbox is None or self.locked_grasp is None:
            return annotated

        x1, y1, x2, y2 = [int(v) for v in self.locked_bbox]
        gx, gy = [int(v) for v in self.locked_grasp]
        sy = int(self.locked_stop_y) if self.locked_stop_y is not None else y2

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 200, 0), 2)
        cv2.circle(annotated, (gx, gy), 8, (0, 255, 255), -1)
        cv2.line(annotated, (x1, sy), (x2, sy), (0, 0, 255), 2)

        label = f"{status_text} conf={self.locked_conf:.2f}"
        if self.locked_depth_m is not None:
            label += f" depth={self.locked_depth_m:.3f}m"

        cv2.putText(
            annotated,
            label,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return annotated

    def _extract_candidate(self, det_row, proto_single, frame_shape, dwdh):
        x1, y1, x2, y2 = map(int, det_row[:4].tolist())
        conf = float(det_row[4].item())
        cls_id = int(det_row[5].item())
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
        else:
            cx = float(np.mean(xs))
            cy = float(np.mean(ys))

            dist = cv2.distanceTransform(mask_clean, cv2.DIST_L2, 5)
            _, _, _, max_loc = cv2.minMaxLoc(dist)
            interior_x = float(max_loc[0])
            interior_y = float(max_loc[1])

            thickest = self._select_grasp_from_thickest_band(mask_clean, xs, ys)
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

            if (
                grasp_y < 0 or grasp_y >= mask_clean.shape[0]
                or grasp_x < 0 or grasp_x >= mask_clean.shape[1]
                or mask_clean[grasp_y, grasp_x] == 0
            ):
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
            "bbox": (x1, y1, x2, y2),
            "conf": conf,
            "cls_id": cls_id,
            "grasp_x": grasp_x,
            "grasp_y": grasp_y,
            "stop_y": stop_y,
            "depth_m": depth_m,
            "mask_bin": mask_for_overlay,
            "row_width_px": row_width_px,
            "thickest_row_y": thickest_row_y,
            "thickest_left": thickest_left,
            "thickest_right": thickest_right,
            "band_top": band_top,
            "band_bottom": band_bottom,
        }

    def _select_grasp_from_thickest_band(self, mask_clean: np.ndarray, xs: np.ndarray, ys: np.ndarray):
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
                best = (score, row_center_x, float(row_y), row_width, row_left, row_right)

        if best is None:
            return None

        _, row_center_x, row_y, row_width, row_left, row_right = best
        return {
            "x": float(row_center_x),
            "y": float(row_y),
            "row_width": int(row_width),
            "row_left": int(row_left),
            "row_right": int(row_right),
            "band_top": int(band_top),
            "band_bottom": int(band_bottom),
        }

    def _select_candidate(self, candidates):
        if not candidates:
            return None

        if not self.locked or self.locked_bbox is None:
            return max(candidates, key=lambda c: c["conf"])

        locked_bbox = self.locked_bbox
        locked_cx, locked_cy = bbox_center(*locked_bbox)

        matched = []
        unmatched = []

        for c in candidates:
            cx, cy = bbox_center(*c["bbox"])
            dist = float(np.hypot(cx - locked_cx, cy - locked_cy))
            iou = bbox_iou(c["bbox"], locked_bbox)

            c["match_dist"] = dist
            c["match_iou"] = iou

            if dist <= self.match_center_dist_px or iou >= self.min_iou_for_match:
                matched.append(c)
            else:
                unmatched.append(c)

        if matched:
            matched.sort(key=lambda c: (c["match_dist"], -c["conf"]))
            return matched[0]

        best_unmatched = max(unmatched, key=lambda c: c["conf"]) if unmatched else None
        if best_unmatched is None:
            return None

        if not self.prefer_locked_target:
            return best_unmatched

        if best_unmatched["conf"] >= (self.locked_conf + self.switch_conf_margin):
            return best_unmatched

        return None

    def cb_image(self, msg: Image):
        if not self.enabled:
            return

        try:
            frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            annotated = frame_bgr.copy()
            now = time.time()

            img_tensor, _, dwdh = preprocess_bgr(frame_bgr, imgsz=IMGSZ, device=DEVICE)
            outputs = self._predict(img_tensor)
            pred, proto = pick_pred_and_proto(outputs)

            if now - self.last_shape_log_time > 2.0:
                self.get_logger().info(
                    f"pred shape={tuple(pred.shape)}, proto shape={tuple(proto.shape)}"
                )
                self.last_shape_log_time = now

            if pred.ndim != 3:
                raise RuntimeError(f"Unexpected pred ndim: {pred.ndim}, shape={tuple(pred.shape)}")

            if pred.shape[1] == 38 and pred.shape[2] != 38:
                pred = pred.transpose(1, 2)

            pred = pred[0]

            if pred.ndim != 2 or pred.shape[1] != 38:
                raise RuntimeError(f"Unexpected pred shape after normalization: {tuple(pred.shape)}")

            confs = pred[:, 4]
            keep = confs > self.conf_threshold
            det = pred[keep]

            if det.shape[0] > 0:
                det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], frame_bgr.shape).round()
                cls_keep = det[:, 5] == 0
                det = det[cls_keep]

            candidates = []
            if det.shape[0] > 0:
                proto_single = proto[0]
                for i in range(det.shape[0]):
                    c = self._extract_candidate(det[i], proto_single, frame_bgr.shape, dwdh)
                    if c["depth_m"] is None:
                        continue
                    candidates.append(c)

            selected = self._select_candidate(candidates)

            # Require high confidence only when creating/reacquiring a lock
            if selected is not None and (not self.locked):
                if selected["conf"] < self.lock_conf_threshold:
                    if now - self.last_lock_log_time > 0.5:
                        self.get_logger().info(
                            f"Best candidate conf={selected['conf']:.2f} below "
                            f"lock threshold {self.lock_conf_threshold:.2f}; not locking"
                        )
                        self.last_lock_log_time = now
                    selected = None

            if selected is not None:
                smoothed = self._smooth(
                    selected["grasp_x"],
                    selected["grasp_y"],
                    selected["stop_y"],
                    selected["conf"],
                    allow_large_jump=not self.locked,
                )

                if smoothed is not None:
                    gx, gy, sy, cf = smoothed

                    self.locked = True
                    self.locked_bbox = selected["bbox"]
                    self.locked_grasp = (gx, gy)
                    self.locked_stop_y = sy
                    self.locked_conf = cf
                    self.locked_depth_m = selected["depth_m"]
                    self.last_seen_time = now

                    self._publish_target(gx, gy, sy, cf, self.locked_bbox)

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

                    annotated = self._draw_locked_state(annotated, status_text="LOCKED")

                    if now - self.last_det_log_time > 0.5:
                        x1, y1, x2, y2 = self.locked_bbox
                        self.get_logger().info(
                            f"LOCKED TARGET: bbox=({x1},{y1},{x2},{y2}) "
                            f"conf={self.locked_conf:.2f} "
                            f"grasp=({gx:.1f},{gy:.1f}) "
                            f"stop_y={sy:.1f} depth={self.locked_depth_m:.3f} "
                            f"row_width={selected['row_width_px']} thick_row_y={selected['thickest_row_y']}"
                        )
                        self.last_det_log_time = now

                    self._publish_debug(msg, annotated)
                    return

            if self.locked and self.hold_last_target:
                age = now - self.last_seen_time
                if age <= self.lock_timeout_sec:
                    gx, gy = self.locked_grasp
                    sy = self.locked_stop_y
                    cf = self.locked_conf

                    self._publish_target(gx, gy, sy, cf, self.locked_bbox)

                    annotated = self._draw_locked_state(
                        annotated, status_text=f"HOLD {age:.2f}s"
                    )

                    if now - self.last_lock_log_time > 0.5:
                        self.get_logger().info(
                            f"Holding locked target for {age:.2f}s after detection dropout"
                        )
                        self.last_lock_log_time = now

                    self._publish_debug(msg, annotated)
                    return

            if self.locked:
                self.get_logger().info("Lock timed out, clearing target")
                self._clear_lock()

            cv2.putText(
                annotated,
                "No sock locked",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            self._publish_debug(msg, annotated)

        except Exception as e:
            self.get_logger().error(f"Inference callback failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = SockMaskTargetTRTNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()