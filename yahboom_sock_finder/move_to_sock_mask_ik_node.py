#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time
from collections import deque
from statistics import median

import numpy as np
import rclpy
import transforms3d as tfs
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup

from arm_interface.srv import ArmKinemarics
from arm_msgs.msg import ArmJoints
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Bool, Float32, Float32MultiArray


class GraspSockIKNode(Node):
    def __init__(self):
        super().__init__("grasp_sock_ik_node")

        self.cb_main = ReentrantCallbackGroup()
        self.cb_srv = MutuallyExclusiveCallbackGroup()
        self.cb_timer = MutuallyExclusiveCallbackGroup()
        # -------------------------------------------------
        # Parameters
        # -------------------------------------------------
        self.declare_parameter("scan_found_topic", "/sock/scan_found")
        self.declare_parameter("mask_target_topic", "/sock/mask_target")
        self.declare_parameter("depth_topic", "/camera/depth/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/depth/camera_info")
        self.declare_parameter("arm_topic", "/arm6_joints")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("done_topic", "/sock/grasp_ik_done")
        self.declare_parameter("distance_topic", "/sock/grasp_depth_distance_m")
        self.declare_parameter("ik_service_name", "/get_kinemarics")
        self.declare_parameter("grasp_attempt_topic", "/sock/grasp_attempt")
        self.declare_parameter("retry_request_topic", "/sock/grasp_retry_requested")
        self.declare_parameter("grasp_complete_topic", "/sock/grasp_complete")
        self.declare_parameter("bbox_topic", "/sock/best_bbox")

        # Startup pose: kept in params, but state-machine use is commented out for now
        self.declare_parameter("startup_pose_enabled", True)
        self.declare_parameter("startup_joint1", 90.0)
        self.declare_parameter("startup_joint2", 90.0)
        self.declare_parameter("startup_joint3", 45.0)
        self.declare_parameter("startup_joint4", 0.0)
        self.declare_parameter("startup_joint5", 90.0)
        self.declare_parameter("startup_joint6", 90.0)
        self.declare_parameter("startup_runtime_ms", 1500)
        self.declare_parameter("startup_settle_sec", 0.30)

        # Modular recent-target reacquire behavior during APPROACH
        self.declare_parameter("enable_recent_target_reacquire", True)
        self.declare_parameter("recent_target_reacquire_sec", 1.0)
        self.declare_parameter("recent_target_backup_speed", -0.05)

        self.declare_parameter("kin_name_fk", "fk")
        self.declare_parameter("kin_name_ik", "ik")

        # Start directly from TRT target by default
        self.declare_parameter("require_scan_found", False)
        self.declare_parameter("mask_start_min_confidence", 0.45)
        self.declare_parameter("mask_start_stable_count", 3)

        # Depth estimation
        self.declare_parameter("depth_is_meters", False)
        self.declare_parameter("depth_center_window", 6)
        self.declare_parameter("depth_floor_window", 10)
        self.declare_parameter("depth_floor_probe_offset_px", 18)
        self.declare_parameter("depth_stop_probe_backoff_px", 8)
        self.declare_parameter("prefer_farther_depth", True)
        self.declare_parameter("depth_history_len", 5)
        self.declare_parameter("min_valid_distance_m", 0.08)
        self.declare_parameter("max_valid_distance_m", 3.0)

        self.declare_parameter("min_confidence", 0.35)
        self.declare_parameter("target_timeout_sec", 0.5)
        self.declare_parameter("target_history_len", 5)

        # Base approach
        self.declare_parameter("target_center_x", 320.0)
        self.declare_parameter("stop_line_target_y", 448.0)
        self.declare_parameter("final_center_deadband_x_px", 14.0)  # loosened from 8.0
        self.declare_parameter("stop_deadband_y_px", 4.0)
        self.declare_parameter("approach_turn_only_px", 110.0)
        self.declare_parameter("angular_gain", 0.0035)
        self.declare_parameter("max_angular_speed", 0.18)
        self.declare_parameter("linear_gain_from_stop_y", 0.0016)
        self.declare_parameter("min_linear_speed", 0.012)
        self.declare_parameter("max_linear_speed", 0.040)
        self.declare_parameter("desired_grasp_distance_m", 0.285)
        self.declare_parameter("grasp_distance_deadband_m", 0.015)
        self.declare_parameter("enable_base_approach", True)

        # These remain declared but are not used by the restored old approach logic
        self.declare_parameter("allow_stopline_reach_trigger", True)
        self.declare_parameter("allow_depth_reach_trigger", False)
        self.declare_parameter("stopline_reach_margin_px", 0.0)
        self.declare_parameter("rotate_only_when_reachable", True)

        # Stability before freezing target
        self.declare_parameter("freeze_depth_tolerance_m", 0.010)
        self.declare_parameter("freeze_u_tolerance_px", 12.0)  # loosened from 8.0
        self.declare_parameter("freeze_v_tolerance_px", 8.0)
        self.declare_parameter("freeze_samples_required", 4)

        # Commit sooner once target is good enough instead of endlessly micro-readjusting
        self.declare_parameter("approach_commit_samples_required", 2)
        self.declare_parameter("approach_commit_center_margin_px", 18.0)
        self.declare_parameter("approach_commit_depth_margin_m", 0.028)
        self.declare_parameter("use_ik_commit_check", True)
        self.declare_parameter("ik_commit_x_margin_px", 28.0)
        self.declare_parameter("ik_commit_depth_margin_m", 0.055)

        # Bias the TRT yellow-dot target a little above the dot so the gripper overshoots upward
        # in image space instead of landing below the sock.
        self.declare_parameter("grasp_target_u_bias_px", 0.0)
        self.declare_parameter("grasp_target_v_overshoot_px", 12.0)

        # Sideways / horizontal sock handling:
        # if bbox width is meaningfully larger than bbox height, treat the sock as sideways
        # and overshoot farther so the gripper reaches deeper onto the sock.
        self.declare_parameter("bbox_timeout_sec", 0.50)
        self.declare_parameter("bbox_min_confidence", 0.35)
        self.declare_parameter("sideways_aspect_ratio_threshold", 1.25)
        self.declare_parameter("sideways_extra_v_overshoot_px", 10.0)
        self.declare_parameter("sideways_pregrasp_x_extra_m", -0.008)
        self.declare_parameter("sideways_grasp_x_extra_m", -0.012)
        self.declare_parameter("sideways_touch_x_extra_m", -0.022)

        # Camera intrinsics defaults
        self.declare_parameter("fx", 477.57)
        self.declare_parameter("fy", 477.56)
        self.declare_parameter("cx", 319.38)
        self.declare_parameter("cy", 238.64)

        # End-effector to camera transform
        self.declare_parameter("end_to_cam_r00", 0.0)
        self.declare_parameter("end_to_cam_r01", 0.0)
        self.declare_parameter("end_to_cam_r02", 1.0)
        self.declare_parameter("end_to_cam_r10", -1.0)
        self.declare_parameter("end_to_cam_r11", 0.0)
        self.declare_parameter("end_to_cam_r12", 0.0)
        self.declare_parameter("end_to_cam_r20", 0.0)
        self.declare_parameter("end_to_cam_r21", -1.0)
        self.declare_parameter("end_to_cam_r22", 0.0)
        self.declare_parameter("end_to_cam_tx", -0.1000)
        self.declare_parameter("end_to_cam_ty", 0.0)
        self.declare_parameter("end_to_cam_tz", 0.0482)

        # Global world-space offsets
        self.declare_parameter("x_offset", 0.005)
        self.declare_parameter("y_offset", 0.012813175910359572)
        self.declare_parameter("z_offset", -0.008798902659958995)

        # Phase offsets
        self.declare_parameter("pregrasp_x_offset_m", -0.084)
        self.declare_parameter("pregrasp_y_offset_m", -0.019)
        self.declare_parameter("pregrasp_z_offset_m", 0.020)

        self.declare_parameter("grasp_x_offset_m", -0.084)
        self.declare_parameter("grasp_y_offset_m", -0.019)
        self.declare_parameter("grasp_z_offset_m", -0.040)

        # Restored from old code
        self.declare_parameter("touch_x_offset_m", 0)
        self.declare_parameter("touch_y_offset_m", -0.019)
        self.declare_parameter("touch_z_offset_m", -0.115)
        self.declare_parameter("touch_floor_extra_drop_m", -0.045)

        self.declare_parameter("lift_x_offset_m", 0.0)
        self.declare_parameter("lift_y_offset_m", 0.0)
        self.declare_parameter("lift_z_offset_m", 0.0)

        # Floor touch Z override
        self.declare_parameter("use_floor_touch_z_override", True)
        self.declare_parameter("floor_touch_z", 0.0007396288712523402)
        self.declare_parameter("touch_surface_margin_m", 0.0)

        # Orientation override
        self.declare_parameter("use_calibrated_touch_orientation", True)

        self.declare_parameter("pregrasp_roll_override", 0.00012579265351099072)
        self.declare_parameter("pregrasp_pitch_override", 1.04719739525615)
        self.declare_parameter("pregrasp_yaw_override", 0.00008416737107432895)

        self.declare_parameter("grasp_roll_override", 0.00012579265351099072)
        self.declare_parameter("grasp_pitch_override", 1.04719739525615)
        self.declare_parameter("grasp_yaw_override", 0.00008416737107432895)

        self.declare_parameter("touch_roll_override", 0.00012579265351099072)
        self.declare_parameter("touch_pitch_override", 1.04719739525615)
        self.declare_parameter("touch_yaw_override", 0.00008416737107432895)

        # Fallback hardcoded orientation
        self.declare_parameter("ik_roll", math.pi)
        self.declare_parameter("ik_pitch", math.pi / 2.0)
        self.declare_parameter("ik_yaw", 0.0)

        self.declare_parameter("stabilize_wrist_joint", True)
        self.declare_parameter("wrist_joint_index", 4)
        self.declare_parameter("desired_wrist_deg", 90.0)

        self.declare_parameter("control_rate_hz", 10.0)
        self.declare_parameter("pregrasp_runtime_ms", 1000)
        self.declare_parameter("grasp_runtime_ms", 1000)
        # Restored from old code
        self.declare_parameter("touch_runtime_ms", 1600)
        self.declare_parameter("gripper_runtime_ms", 500)
        self.declare_parameter("lift_runtime_ms", 1200)
        self.declare_parameter("pregrasp_settle_sec", 0.8)
        self.declare_parameter("grasp_settle_sec", 1.0)
        self.declare_parameter("touch_settle_sec", 1.8)
        self.declare_parameter("close_hold_sec", 0.6)
        self.declare_parameter("lift_settle_sec", 0.7)

        self.declare_parameter("open_joint6_value", 90.0)
        self.declare_parameter("close_joint6_value", 180.0)

        self.declare_parameter("carry_pose_enabled", True)
        self.declare_parameter("carry_joint1", 90.0)
        self.declare_parameter("carry_joint2", 90.0)
        self.declare_parameter("carry_joint3", 90.0)
        self.declare_parameter("carry_joint4", 0.0)
        self.declare_parameter("carry_joint5", 90.0)
        self.declare_parameter("carry_joint6", 180.0)
        self.declare_parameter("carry_runtime_ms", 2000)
        self.declare_parameter("carry_settle_sec", 1.0)

        self.declare_parameter("service_timeout_sec", 5.0)

        gp = lambda name: self.get_parameter(name).value

        self.scan_found_topic = str(gp("scan_found_topic"))
        self.mask_target_topic = str(gp("mask_target_topic"))
        self.depth_topic = str(gp("depth_topic"))
        self.camera_info_topic = str(gp("camera_info_topic"))
        self.arm_topic = str(gp("arm_topic"))
        self.cmd_vel_topic = str(gp("cmd_vel_topic"))
        self.done_topic = str(gp("done_topic"))
        self.distance_topic = str(gp("distance_topic"))
        self.ik_service_name = str(gp("ik_service_name"))
        self.grasp_attempt_topic = str(gp("grasp_attempt_topic"))
        self.retry_request_topic = str(gp("retry_request_topic"))
        self.grasp_complete_topic = str(gp("grasp_complete_topic"))
        self.bbox_topic = str(gp("bbox_topic"))

        self.startup_pose_enabled = bool(gp("startup_pose_enabled"))
        self.startup_joints = [
            float(gp("startup_joint1")),
            float(gp("startup_joint2")),
            float(gp("startup_joint3")),
            float(gp("startup_joint4")),
            float(gp("startup_joint5")),
            float(gp("startup_joint6")),
        ]
        self.startup_runtime_ms = int(gp("startup_runtime_ms"))
        self.startup_settle_sec = float(gp("startup_settle_sec"))

        self.enable_recent_target_reacquire = bool(gp("enable_recent_target_reacquire"))
        self.recent_target_reacquire_sec = float(gp("recent_target_reacquire_sec"))
        self.recent_target_backup_speed = float(gp("recent_target_backup_speed"))

        self.kin_name_fk = str(gp("kin_name_fk"))
        self.kin_name_ik = str(gp("kin_name_ik"))

        self.require_scan_found = bool(gp("require_scan_found"))
        self.mask_start_min_confidence = float(gp("mask_start_min_confidence"))
        self.mask_start_stable_count = int(gp("mask_start_stable_count"))

        self.depth_is_meters = bool(gp("depth_is_meters"))
        self.depth_center_window = int(gp("depth_center_window"))
        self.depth_floor_window = int(gp("depth_floor_window"))
        self.depth_floor_probe_offset_px = int(gp("depth_floor_probe_offset_px"))
        self.depth_stop_probe_backoff_px = int(gp("depth_stop_probe_backoff_px"))
        self.prefer_farther_depth = bool(gp("prefer_farther_depth"))
        self.depth_history_len = int(gp("depth_history_len"))
        self.min_valid_distance_m = float(gp("min_valid_distance_m"))
        self.max_valid_distance_m = float(gp("max_valid_distance_m"))

        self.min_confidence = float(gp("min_confidence"))
        self.target_timeout_sec = float(gp("target_timeout_sec"))
        self.target_history_len = int(gp("target_history_len"))

        self.target_center_x = float(gp("target_center_x"))
        self.stop_line_target_y = float(gp("stop_line_target_y"))
        self.final_center_deadband_x_px = float(gp("final_center_deadband_x_px"))
        self.stop_deadband_y_px = float(gp("stop_deadband_y_px"))
        self.approach_turn_only_px = float(gp("approach_turn_only_px"))
        self.angular_gain = float(gp("angular_gain"))
        self.max_angular_speed = float(gp("max_angular_speed"))
        self.linear_gain_from_stop_y = float(gp("linear_gain_from_stop_y"))
        self.min_linear_speed = float(gp("min_linear_speed"))
        self.max_linear_speed = float(gp("max_linear_speed"))
        self.desired_grasp_distance_m = float(gp("desired_grasp_distance_m"))
        self.grasp_distance_deadband_m = float(gp("grasp_distance_deadband_m"))
        self.enable_base_approach = bool(gp("enable_base_approach"))

        self.allow_stopline_reach_trigger = bool(gp("allow_stopline_reach_trigger"))
        self.allow_depth_reach_trigger = bool(gp("allow_depth_reach_trigger"))
        self.stopline_reach_margin_px = float(gp("stopline_reach_margin_px"))
        self.rotate_only_when_reachable = bool(gp("rotate_only_when_reachable"))

        self.freeze_depth_tolerance_m = float(gp("freeze_depth_tolerance_m"))
        self.freeze_u_tolerance_px = float(gp("freeze_u_tolerance_px"))
        self.freeze_v_tolerance_px = float(gp("freeze_v_tolerance_px"))
        self.freeze_samples_required = int(gp("freeze_samples_required"))
        self.approach_commit_samples_required = int(gp("approach_commit_samples_required"))
        self.approach_commit_center_margin_px = float(gp("approach_commit_center_margin_px"))
        self.approach_commit_depth_margin_m = float(gp("approach_commit_depth_margin_m"))
        self.use_ik_commit_check = bool(gp("use_ik_commit_check"))
        self.ik_commit_x_margin_px = float(gp("ik_commit_x_margin_px"))
        self.ik_commit_depth_margin_m = float(gp("ik_commit_depth_margin_m"))
        self.grasp_target_u_bias_px = float(gp("grasp_target_u_bias_px"))
        self.grasp_target_v_overshoot_px = float(gp("grasp_target_v_overshoot_px"))

        self.bbox_timeout_sec = float(gp("bbox_timeout_sec"))
        self.bbox_min_confidence = float(gp("bbox_min_confidence"))
        self.sideways_aspect_ratio_threshold = float(gp("sideways_aspect_ratio_threshold"))
        self.sideways_extra_v_overshoot_px = float(gp("sideways_extra_v_overshoot_px"))
        self.sideways_pregrasp_x_extra_m = float(gp("sideways_pregrasp_x_extra_m"))
        self.sideways_grasp_x_extra_m = float(gp("sideways_grasp_x_extra_m"))
        self.sideways_touch_x_extra_m = float(gp("sideways_touch_x_extra_m"))

        self.fx = float(gp("fx"))
        self.fy = float(gp("fy"))
        self.cx = float(gp("cx"))
        self.cy = float(gp("cy"))

        r00 = float(gp("end_to_cam_r00"))
        r01 = float(gp("end_to_cam_r01"))
        r02 = float(gp("end_to_cam_r02"))
        r10 = float(gp("end_to_cam_r10"))
        r11 = float(gp("end_to_cam_r11"))
        r12 = float(gp("end_to_cam_r12"))
        r20 = float(gp("end_to_cam_r20"))
        r21 = float(gp("end_to_cam_r21"))
        r22 = float(gp("end_to_cam_r22"))
        tx = float(gp("end_to_cam_tx"))
        ty = float(gp("end_to_cam_ty"))
        tz = float(gp("end_to_cam_tz"))

        self.EndToCamMat = np.array(
            [
                [r00, r01, r02, tx],
                [r10, r11, r12, ty],
                [r20, r21, r22, tz],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        self.x_offset = float(gp("x_offset"))
        self.y_offset = float(gp("y_offset"))
        self.z_offset = float(gp("z_offset"))

        self.pregrasp_x_offset_m = float(gp("pregrasp_x_offset_m"))
        self.pregrasp_y_offset_m = float(gp("pregrasp_y_offset_m"))
        self.pregrasp_z_offset_m = float(gp("pregrasp_z_offset_m"))

        self.grasp_x_offset_m = float(gp("grasp_x_offset_m"))
        self.grasp_y_offset_m = float(gp("grasp_y_offset_m"))
        self.grasp_z_offset_m = float(gp("grasp_z_offset_m"))

        self.touch_x_offset_m = float(gp("touch_x_offset_m"))
        self.touch_y_offset_m = float(gp("touch_y_offset_m"))
        self.touch_z_offset_m = float(gp("touch_z_offset_m"))
        self.touch_floor_extra_drop_m = float(gp("touch_floor_extra_drop_m"))

        self.lift_x_offset_m = float(gp("lift_x_offset_m"))
        self.lift_y_offset_m = float(gp("lift_y_offset_m"))
        self.lift_z_offset_m = float(gp("lift_z_offset_m"))

        self.use_floor_touch_z_override = bool(gp("use_floor_touch_z_override"))
        self.floor_touch_z = float(gp("floor_touch_z"))
        self.touch_surface_margin_m = float(gp("touch_surface_margin_m"))

        self.use_calibrated_touch_orientation = bool(gp("use_calibrated_touch_orientation"))
        self.pregrasp_roll_override = float(gp("pregrasp_roll_override"))
        self.pregrasp_pitch_override = float(gp("pregrasp_pitch_override"))
        self.pregrasp_yaw_override = float(gp("pregrasp_yaw_override"))
        self.grasp_roll_override = float(gp("grasp_roll_override"))
        self.grasp_pitch_override = float(gp("grasp_pitch_override"))
        self.grasp_yaw_override = float(gp("grasp_yaw_override"))
        self.touch_roll_override = float(gp("touch_roll_override"))
        self.touch_pitch_override = float(gp("touch_pitch_override"))
        self.touch_yaw_override = float(gp("touch_yaw_override"))

        self.ik_roll = float(gp("ik_roll"))
        self.ik_pitch = float(gp("ik_pitch"))
        self.ik_yaw = float(gp("ik_yaw"))

        self.stabilize_wrist_joint = bool(gp("stabilize_wrist_joint"))
        self.wrist_joint_index = int(gp("wrist_joint_index"))
        self.desired_wrist_deg = float(gp("desired_wrist_deg"))

        self.control_rate_hz = float(gp("control_rate_hz"))
        self.pregrasp_runtime_ms = int(gp("pregrasp_runtime_ms"))
        self.grasp_runtime_ms = int(gp("grasp_runtime_ms"))
        self.touch_runtime_ms = int(gp("touch_runtime_ms"))
        self.gripper_runtime_ms = int(gp("gripper_runtime_ms"))
        self.lift_runtime_ms = int(gp("lift_runtime_ms"))
        self.pregrasp_settle_sec = float(gp("pregrasp_settle_sec"))
        self.grasp_settle_sec = float(gp("grasp_settle_sec"))
        self.touch_settle_sec = float(gp("touch_settle_sec"))
        self.close_hold_sec = float(gp("close_hold_sec"))
        self.lift_settle_sec = float(gp("lift_settle_sec"))

        self.open_joint6_value = float(gp("open_joint6_value"))
        self.close_joint6_value = float(gp("close_joint6_value"))

        self.carry_pose_enabled = bool(gp("carry_pose_enabled"))
        self.carry_joints = [
            float(gp("carry_joint1")),
            float(gp("carry_joint2")),
            float(gp("carry_joint3")),
            float(gp("carry_joint4")),
            float(gp("carry_joint5")),
            float(gp("carry_joint6")),
        ]
        self.carry_runtime_ms = int(gp("carry_runtime_ms"))
        self.carry_settle_sec = float(gp("carry_settle_sec"))

        self.service_timeout_sec = float(gp("service_timeout_sec"))

        # Service-call protection / caching
        self.fk_busy = False
        self.ik_busy = False
        self.last_fk_xyz = None
        self.last_fk_rpy = None
        self.last_fk_attempt_time = 0.0
        self.last_ik_attempt_time = 0.0
        self.fk_min_interval_sec = 0.20
        self.ik_min_interval_sec = 0.25

        self.bridge = CvBridge()

        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.arm_pub = self.create_publisher(ArmJoints, self.arm_topic, 10)
        self.done_pub = self.create_publisher(Bool, self.done_topic, 10)
        self.distance_pub = self.create_publisher(Float32, self.distance_topic, 10)
        self.grasp_attempt_pub = self.create_publisher(Bool, self.grasp_attempt_topic, 10)
        self.grasp_complete_pub = self.create_publisher(Bool, self.grasp_complete_topic, 10)
        self.override_pub = self.create_publisher(Bool, "/sock/patrol_override_active", 10)

        self.create_subscription(Bool, self.scan_found_topic, self.scan_found_callback, 10, callback_group=self.cb_main)
        self.create_subscription(Float32MultiArray, self.mask_target_topic, self.mask_target_callback, 10, callback_group=self.cb_main)
        self.create_subscription(Float32MultiArray, self.bbox_topic, self.bbox_callback, 10, callback_group=self.cb_main)
        self.create_subscription(Image, self.depth_topic, self.depth_callback, 10, callback_group=self.cb_main)
        self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, 10, callback_group=self.cb_main)
        self.create_subscription(ArmJoints, self.arm_topic, self.arm_pose_callback, 10, callback_group=self.cb_main)

        self.create_subscription(Bool, self.retry_request_topic, self.retry_request_callback, 10, callback_group=self.cb_main)

        self.kin_client = self.create_client(ArmKinemarics, self.ik_service_name, callback_group=self.cb_srv)

        self.scan_found = False
        self.latest_target = None
        self.last_target_time = None
        self.latest_bbox = None
        self.last_bbox_time = None
        self.latest_depth = None
        self.camera_info_received = False

        self.mask_seen_stable_count = 0

        self.grasp_x_hist = deque(maxlen=self.target_history_len)
        self.grasp_y_hist = deque(maxlen=self.target_history_len)
        self.stop_y_hist = deque(maxlen=self.target_history_len)
        self.depth_hist = deque(maxlen=self.depth_history_len)
        self.ready_samples = deque(maxlen=max(self.freeze_samples_required, 3))

        self.current_joints = list(self.startup_joints)
        self.last_pregrasp_joints = None
        self.last_grasp_joints = None
        self.last_touch_joints = None
        self.last_closed_grasp_joints = None
        self.last_lift_joints = None

        # Startup pose section re-enabled so retry/handoff resets can visibly
        # return the arm to the known startup pose before scanning again.
        self.state = "STARTUP_POSE" if self.startup_pose_enabled else "WAIT_FOR_SCAN"
        self.state_start_time = time.time()
        self.frozen_world_point = None
        self.grasp_started = False
        self.gripper_locked_closed = False

        self.post_lift_reinforced = False
        self.carry_reinforced = False
        self.reacquire_backup_started_time = None
        self.startup_pose_sent = False
        self.handoff_to_carry_active = False
        self.pending_patrol_release_after_startup = False
        self.approach_ready_count = 0
        self.last_adjusted_target = None

        timer_period = 1.0 / max(self.control_rate_hz, 1.0)
        self.timer = self.create_timer(timer_period, self.timer_callback, callback_group=self.cb_timer)
        self.get_logger().warn("################ SOCK PICKUP DEFAULTS BAKED IN ################")
        self.get_logger().warn(
            f"DEFAULTS: require_scan_found={self.require_scan_found}, "
            f"use_calibrated_touch_orientation={self.use_calibrated_touch_orientation}, "
            f"use_floor_touch_z_override={self.use_floor_touch_z_override}"
        )
        self.get_logger().warn(
            f"DEFAULT TOUCH RPY=({self.touch_roll_override:.6f}, "
            f"{self.touch_pitch_override:.6f}, {self.touch_yaw_override:.6f})"
        )
        self.get_logger().warn(
            f"DEFAULT FLOOR Z={self.floor_touch_z:.6f}, "
            f"XY OFFSETS pre/grasp/touch=({self.pregrasp_x_offset_m:.3f}, {self.pregrasp_y_offset_m:.3f}) / "
            f"({self.grasp_x_offset_m:.3f}, {self.grasp_y_offset_m:.3f}) / "
            f"({self.touch_x_offset_m:.3f}, {self.touch_y_offset_m:.3f})"
        )
        self.get_logger().warn(
            f"STARTUP_POSE params kept but startup execution disabled for now: {self.startup_joints}"
        )
        self.get_logger().warn(
            "Restored old grab behavior: touch_x_offset_m=-0.066, touch_runtime_ms=1600, "
            "old approach_with_trt depth+stopline readiness logic"
        )
        self.get_logger().warn(
            "Loosened yellow-dot alignment: final_center_deadband_x_px=14.0, freeze_u_tolerance_px=12.0"
        )
        self.get_logger().warn(
            f"Yellow-dot overshoot enabled: u_bias={self.grasp_target_u_bias_px:.1f}px, "
            f"v_overshoot={self.grasp_target_v_overshoot_px:.1f}px, "
            f"commit_samples={self.approach_commit_samples_required}, use_ik_commit_check={self.use_ik_commit_check}"
        )

    def scan_found_callback(self, msg: Bool) -> None:
        if not self.require_scan_found:
            return
        new_val = bool(msg.data)
        if self.grasp_started:
            return
        if self.scan_found and not new_val:
            self.get_logger().warn("scan_found dropped false before grasp start, resetting")
            self.reset_state_machine()
        self.scan_found = new_val
        if self.scan_found and self.state == "WAIT_FOR_SCAN":
            self.state = "APPROACH"
            self.state_start_time = time.time()
            self.get_logger().info("Scan handoff received. Starting APPROACH.")



    def bbox_callback(self, msg: Float32MultiArray) -> None:
        if len(msg.data) < 5:
            return
        self.latest_bbox = [float(v) for v in msg.data[:5]]
        self.last_bbox_time = time.time()

    def mask_target_callback(self, msg: Float32MultiArray) -> None:
        if len(msg.data) < 4:
            self.mask_seen_stable_count = 0
            return
        grasp_x = float(msg.data[0])
        grasp_y = float(msg.data[1])
        stop_y = float(msg.data[2])
        conf = float(msg.data[3])
        if conf < self.min_confidence:
            self.mask_seen_stable_count = 0
            return
        self.latest_target = (grasp_x, grasp_y, stop_y, conf)
        self.last_target_time = time.time()
        self.grasp_x_hist.append(grasp_x)
        self.grasp_y_hist.append(grasp_y)
        self.stop_y_hist.append(stop_y)
        if conf >= self.mask_start_min_confidence:
            self.mask_seen_stable_count += 1
        else:
            self.mask_seen_stable_count = 0
        if (
            not self.require_scan_found
            and not self.grasp_started
            and self.state == "WAIT_FOR_SCAN"
            and self.mask_seen_stable_count >= self.mask_start_stable_count
        ):
            self.scan_found = True
            self.state = "APPROACH"
            self.state_start_time = time.time()
            self.get_logger().info(
                f"TRT_DIRECT_START: stable mask target acquired "
                f"(count={self.mask_seen_stable_count}, conf={conf:.2f}) -> APPROACH"
            )

    def depth_callback(self, msg: Image) -> None:
        try:
            if self.depth_is_meters:
                self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
            else:
                self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1")
        except Exception as exc:
            self.get_logger().error(f"Depth conversion failed: {exc}")

    def camera_info_callback(self, msg: CameraInfo) -> None:
        if len(msg.k) >= 9:
            self.fx = float(msg.k[0])
            self.fy = float(msg.k[4])
            self.cx = float(msg.k[2])
            self.cy = float(msg.k[5])
            self.camera_info_received = True

    def arm_pose_callback(self, msg: ArmJoints) -> None:
        self.current_joints = [
            float(msg.joint1),
            float(msg.joint2),
            float(msg.joint3),
            float(msg.joint4),
            float(msg.joint5),
            float(msg.joint6),
        ]

    @staticmethod
    def clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def command_finished(self, runtime_ms: int, extra_sec: float = 0.20) -> bool:
        return (time.time() - self.state_start_time) >= (float(runtime_ms) / 1000.0 + extra_sec)

    def force_joint6_policy(self, joints):
        out = list(joints)
        while len(out) < 6:
            out.append(90.0)
        if self.gripper_locked_closed:
            out[5] = self.close_joint6_value
        return out

    def publish_done(self, value: bool) -> None:
        msg = Bool()
        msg.data = bool(value)
        self.done_pub.publish(msg)

    def publish_grasp_attempt(self, value: bool) -> None:
        msg = Bool()
        msg.data = bool(value)
        self.grasp_attempt_pub.publish(msg)
        self.get_logger().info(f"GRASP_ATTEMPT_PUB: {msg.data}")

    def publish_grasp_complete(self, value: bool) -> None:
        msg = Bool()
        msg.data = bool(value)
        self.grasp_complete_pub.publish(msg)
        self.get_logger().info(f"GRASP_COMPLETE_PUB: {msg.data}")

    def publish_patrol_override(self, value: bool) -> None:
        msg = Bool()
        msg.data = bool(value)
        self.override_pub.publish(msg)
        self.get_logger().info(f"PATROL_OVERRIDE_PUB: {msg.data}")

    def retry_request_callback(self, msg: Bool) -> None:
        if not bool(msg.data):
            return
        self.get_logger().warn("RETRY_REQUEST received -> resetting grasp state machine and re-running startup pose")
        self.publish_stop()
        self.publish_grasp_attempt(False)
        self.publish_grasp_complete(False)
        self.reset_state_machine()
        self.pending_patrol_release_after_startup = True
        self.state = "STARTUP_POSE" if self.startup_pose_enabled else "WAIT_FOR_SCAN"
        self.state_start_time = time.time()

    def publish_stop(self) -> None:
        self.cmd_pub.publish(Twist())

    def publish_recent_target_backup(self) -> None:
        cmd = Twist()
        cmd.linear.x = float(self.recent_target_backup_speed)
        self.cmd_pub.publish(cmd)

    def should_recent_target_reacquire(self) -> bool:
        if not self.enable_recent_target_reacquire:
            return False
        if self.last_target_time is None:
            return False
        age = time.time() - self.last_target_time
        return age <= self.recent_target_reacquire_sec

    def publish_arm_joints(self, joints, runtime_ms=1200, label="ARM_CMD") -> None:
        out = self.force_joint6_policy(joints)
        msg = ArmJoints()
        msg.joint1 = int(round(out[0]))
        msg.joint2 = int(round(out[1]))
        msg.joint3 = int(round(out[2]))
        msg.joint4 = int(round(out[3]))
        msg.joint5 = int(round(out[4]))
        msg.joint6 = int(round(out[5]))
        msg.time = int(runtime_ms)
        self.arm_pub.publish(msg)
        self.get_logger().info(
            f"{label}: joints="
            f"{[msg.joint1, msg.joint2, msg.joint3, msg.joint4, msg.joint5, msg.joint6]} "
            f"runtime_ms={runtime_ms}"
        )

    def lock_gripper_closed(self):
        self.gripper_locked_closed = True

    def reset_state_machine(self) -> None:
        self.state = "STARTUP_POSE" if self.startup_pose_enabled else "WAIT_FOR_SCAN"
        self.state_start_time = time.time()
        self.frozen_world_point = None
        self.grasp_started = False
        self.gripper_locked_closed = False
        self.scan_found = False
        self.mask_seen_stable_count = 0
        self.last_pregrasp_joints = None
        self.last_grasp_joints = None
        self.last_touch_joints = None
        self.last_closed_grasp_joints = None
        self.last_lift_joints = None
        self.grasp_x_hist.clear()
        self.grasp_y_hist.clear()
        self.stop_y_hist.clear()
        self.depth_hist.clear()
        self.ready_samples.clear()
        self.post_lift_reinforced = False
        self.carry_reinforced = False
        self.reacquire_backup_started_time = None
        self.approach_ready_count = 0
        self.last_adjusted_target = None
        self.publish_stop()
        self.publish_done(False)
        self.publish_grasp_attempt(False)
        self.handoff_to_carry_active = False
        self.pending_patrol_release_after_startup = False

    def target_is_fresh(self) -> bool:
        return self.last_target_time is not None and (
            time.time() - self.last_target_time <= self.target_timeout_sec
        )

    def bbox_is_fresh(self) -> bool:
        if self.latest_bbox is None or self.last_bbox_time is None:
            return False
        if (time.time() - self.last_bbox_time) > self.bbox_timeout_sec:
            return False
        if len(self.latest_bbox) < 5:
            return False
        return float(self.latest_bbox[4]) >= self.bbox_min_confidence

    def get_bbox_width_height(self):
        if not self.bbox_is_fresh():
            return None, None
        return float(self.latest_bbox[2]), float(self.latest_bbox[3])

    def sock_is_sideways(self) -> bool:
        bw, bh = self.get_bbox_width_height()
        if bw is None or bh is None or bh <= 1.0:
            return False
        aspect = bw / bh
        sideways = aspect >= self.sideways_aspect_ratio_threshold
        if sideways:
            self.get_logger().info(
                f"SOCK_SIDEWAYS_DETECTED: bbox_w={bw:.1f}, bbox_h={bh:.1f}, aspect={aspect:.2f}"
            )
        return sideways

    def get_dynamic_v_overshoot_px(self) -> float:
        overshoot = self.grasp_target_v_overshoot_px
        if self.sock_is_sideways():
            overshoot += self.sideways_extra_v_overshoot_px
        return overshoot

    def get_dynamic_phase_x_extra(self, phase_name: str) -> float:
        if not self.sock_is_sideways():
            return 0.0
        if phase_name == "pregrasp":
            return self.sideways_pregrasp_x_extra_m
        if phase_name == "grasp":
            return self.sideways_grasp_x_extra_m
        if phase_name == "touch":
            return self.sideways_touch_x_extra_m
        return 0.0

    def get_patch_depth_median(self, u: int, v: int, radius: int):
        if self.latest_depth is None:
            return None
        depth = self.latest_depth
        h, w = depth.shape[:2]
        u = int(np.clip(u, 0, w - 1))
        v = int(np.clip(v, 0, h - 1))
        r = max(1, int(radius))
        x1 = max(0, u - r)
        x2 = min(w, u + r + 1)
        y1 = max(0, v - r)
        y2 = min(h, v + r + 1)
        patch = depth[y1:y2, x1:x2]
        vals = patch[np.isfinite(patch)]
        if vals.size == 0:
            return None
        if self.depth_is_meters:
            vals = vals[(vals > self.min_valid_distance_m) & (vals < self.max_valid_distance_m)]
            if vals.size == 0:
                return None
            vals = np.sort(vals.astype(np.float64))
            lo = int(len(vals) * 0.15)
            hi = max(lo + 1, int(len(vals) * 0.85))
            vals = vals[lo:hi]
            return float(np.median(vals))
        vals = vals[(vals > 1) & (vals < int(self.max_valid_distance_m * 1000.0))]
        if vals.size == 0:
            return None
        vals = np.sort(vals.astype(np.float64))
        lo = int(len(vals) * 0.15)
        hi = max(lo + 1, int(len(vals) * 0.85))
        vals = vals[lo:hi]
        return float(np.median(vals)) / 1000.0

    def get_target_depth_estimate(self, u: int, v: int, stop_y: float):
        center_depth = self.get_patch_depth_median(u, v, self.depth_center_window)
        floor_probe_v = int(max(v + self.depth_floor_probe_offset_px, stop_y - self.depth_stop_probe_backoff_px))
        floor_depth = self.get_patch_depth_median(u, floor_probe_v, self.depth_floor_window)
        candidates = [d for d in [center_depth, floor_depth] if d is not None]
        if not candidates:
            return None
        chosen = max(candidates) if self.prefer_farther_depth else float(sum(candidates) / len(candidates))
        self.get_logger().info(
            f"DEPTH_ESTIMATE: u={u} v={v} stop_y={stop_y:.1f} "
            f"center_depth={center_depth} floor_depth={floor_depth} chosen={chosen}"
        )
        return float(chosen)

    def get_smoothed_target(self):
        if self.latest_target is None or not self.target_is_fresh():
            return None
        if not self.grasp_x_hist or not self.grasp_y_hist or not self.stop_y_hist:
            return None
        raw_u = float(median(self.grasp_x_hist))
        raw_v = float(median(self.grasp_y_hist))
        stop_y = float(median(self.stop_y_hist))
        _, _, _, conf = self.latest_target

        # Bias the final grasp point slightly above the TRT yellow dot so the arm does not undershoot below it.
        # If the sock looks sideways (wider than tall), overshoot more aggressively.
        dynamic_v_overshoot_px = self.get_dynamic_v_overshoot_px()
        u = raw_u + self.grasp_target_u_bias_px
        v = raw_v - dynamic_v_overshoot_px
        if self.latest_depth is not None:
            h, w = self.latest_depth.shape[:2]
            u = float(np.clip(u, 0, w - 1))
            v = float(np.clip(v, 0, h - 1))

        depth = self.get_target_depth_estimate(int(u), int(v), stop_y)
        if depth is None:
            return None
        self.depth_hist.append(depth)
        depth = float(median(self.depth_hist))
        self.ready_samples.append((u, v, stop_y, depth, time.time()))
        self.last_adjusted_target = (u, v, stop_y, float(conf), depth, raw_u, raw_v)
        dist_msg = Float32()
        dist_msg.data = depth
        self.distance_pub.publish(dist_msg)
        return u, v, stop_y, float(conf), depth

    def target_is_stable_for_freeze(self):
        if len(self.ready_samples) < self.freeze_samples_required:
            return False
        recent = list(self.ready_samples)[-self.freeze_samples_required:]
        us = [x[0] for x in recent]
        vs = [x[1] for x in recent]
        ds = [x[3] for x in recent]
        return (
            (max(us) - min(us)) <= self.freeze_u_tolerance_px
            and (max(vs) - min(vs)) <= self.freeze_v_tolerance_px
            and (max(ds) - min(ds)) <= self.freeze_depth_tolerance_m
        )

    def approach_ready_by_hysteresis(self, centered_x: bool, close_by_depth: bool) -> bool:
        if centered_x and close_by_depth:
            self.approach_ready_count += 1
        else:
            self.approach_ready_count = 0
        return self.approach_ready_count >= self.approach_commit_samples_required

    def pregrasp_ik_reachable(self, u: float, v: float, depth: float) -> bool:
        if not self.use_ik_commit_check:
            return False
        world_xyz = self.compute_world_target(u, v, depth)
        if world_xyz is None:
            return False
        try:
            px, py, pz = [float(val) for val in world_xyz]
            tx = px + self.pregrasp_x_offset_m
            ty = py + self.pregrasp_y_offset_m
            tz = pz + self.pregrasp_z_offset_m
            roll, pitch, yaw = self.get_target_orientation("pregrasp")
            joints = self.solve_ik_blocking(tx, ty, tz, roll, pitch, yaw)
            if joints is None:
                self.get_logger().info("IK_COMMIT_CHECK: no IK solution yet")
                return False
            self.get_logger().info(
                f"IK_COMMIT_CHECK: reachable pregrasp for adjusted target at x={tx:.4f}, y={ty:.4f}, z={tz:.4f}"
            )
            return True
        except Exception as exc:
            self.get_logger().warn(f"IK_COMMIT_CHECK failed: {exc}")
            return False

    def approach_with_trt(self, u: float, v: float, stop_y: float, depth: float) -> bool:
        # Use the TRT yellow dot as a guide, but bias slightly above it and commit once the
        # arm is already in a kinematically good-enough pickup state instead of endlessly readjusting.
        dx = u - self.target_center_x
        stop_err = self.stop_line_target_y - stop_y
        depth_err = depth - self.desired_grasp_distance_m
        centered_x = abs(dx) <= self.final_center_deadband_x_px
        close_by_depth = abs(depth_err) <= self.grasp_distance_deadband_m

        soft_centered_x = abs(dx) <= self.approach_commit_center_margin_px
        soft_close_by_depth = abs(depth_err) <= self.approach_commit_depth_margin_m
        hysteresis_ready = self.approach_ready_by_hysteresis(soft_centered_x, soft_close_by_depth)

        ik_ready = False
        if abs(dx) <= self.ik_commit_x_margin_px and abs(depth_err) <= self.ik_commit_depth_margin_m:
            ik_ready = self.pregrasp_ik_reachable(u, v, depth)

        ready_to_grasp = (
            (centered_x and close_by_depth and (stop_err <= max(self.stop_deadband_y_px, 8.0)))
            or hysteresis_ready
            or ik_ready
        )
        if ready_to_grasp:
            self.publish_stop()
            self.get_logger().info(
                f"APPROACH_READY: u={u:.1f} v={v:.1f} stop_y={stop_y:.1f} depth={depth:.3f} "
                f"dx={dx:.1f} stop_err={stop_err:.1f} depth_err={depth_err:.3f} "
                f"hysteresis_ready={hysteresis_ready} ik_ready={ik_ready} ready_count={self.approach_ready_count}"
            )
            return True

        cmd = Twist()
        cmd.angular.z = self.clamp(-dx * self.angular_gain, -self.max_angular_speed, self.max_angular_speed)
        too_far_by_depth = depth > (self.desired_grasp_distance_m + self.grasp_distance_deadband_m)
        too_far_by_stop = stop_err > self.stop_deadband_y_px
        linear = 0.0
        if abs(dx) <= self.approach_turn_only_px:
            if too_far_by_depth:
                linear = self.min_linear_speed
            if too_far_by_stop:
                linear = max(linear, stop_err * self.linear_gain_from_stop_y)
        elif abs(dx) <= (self.approach_turn_only_px * 1.5):
            if too_far_by_depth:
                linear = self.min_linear_speed
        cmd.linear.x = self.clamp(linear, 0.0, self.max_linear_speed)
        if not self.enable_base_approach:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)
        self.get_logger().info(
            f"APPROACH: u={u:.1f} v={v:.1f} stop_y={stop_y:.1f} depth={depth:.3f} "
            f"dx={dx:.1f} stop_err={stop_err:.1f} depth_err={depth_err:.3f} "
            f"linear={cmd.linear.x:.3f} angular={cmd.angular.z:.3f} "
            f"soft_centered={soft_centered_x} soft_depth={soft_close_by_depth} ready_count={self.approach_ready_count}"
        )
        return False

    def pixel_to_camera_depth(self, pixel_xy, depth_m):
        u, v = pixel_xy
        z = float(depth_m)
        x = (float(u) - self.cx) * z / self.fx
        y = (float(v) - self.cy) * z / self.fy
        return np.array([x, y, z], dtype=np.float64)

    @staticmethod
    def xyz_euler_to_mat(translation_xyz, euler_rpy):
        mat = np.eye(4, dtype=np.float64)
        rot = tfs.euler.euler2mat(float(euler_rpy[0]), float(euler_rpy[1]), float(euler_rpy[2]), axes="sxyz")
        mat[:3, :3] = rot
        mat[:3, 3] = np.array(translation_xyz, dtype=np.float64)
        return mat

    @staticmethod
    def mat_to_xyz_euler(mat):
        t = np.array(mat[:3, 3], dtype=np.float64)
        euler = tfs.euler.mat2euler(mat[:3, :3], axes="sxyz")
        return t, euler

    def _spin_wait_future(self, future, timeout_sec: float) -> bool:
        start = time.time()
        while rclpy.ok() and not future.done():
            if (time.time() - start) > timeout_sec:
                return False
            time.sleep(0.005)
        return future.done()

    def _service_ready(self, label: str, timeout_sec: float = 0.25) -> bool:
        if self.kin_client.service_is_ready():
            return True
        ok = self.kin_client.wait_for_service(timeout_sec=timeout_sec)
        if not ok:
            self.get_logger().warn(f"{label}: IK/FK service not available")
        return ok

    def get_current_end_pose_fk_blocking(self):
        now = time.time()

        if self.fk_busy:
            self.get_logger().warn("FK skipped: previous FK still running")
            if self.last_fk_xyz is not None and self.last_fk_rpy is not None:
                return list(self.last_fk_xyz), list(self.last_fk_rpy)
            return None, None

        if (now - self.last_fk_attempt_time) < self.fk_min_interval_sec:
            if self.last_fk_xyz is not None and self.last_fk_rpy is not None:
                return list(self.last_fk_xyz), list(self.last_fk_rpy)
            return None, None

        if not self._service_ready("FK", timeout_sec=0.25):
            if self.last_fk_xyz is not None and self.last_fk_rpy is not None:
                self.get_logger().warn("FK unavailable; using cached FK pose")
                return list(self.last_fk_xyz), list(self.last_fk_rpy)
            return None, None

        self.last_fk_attempt_time = now
        self.fk_busy = True
        try:
            req = ArmKinemarics.Request()
            req.tar_x = 0.0
            req.tar_y = 0.0
            req.tar_z = 0.0
            req.roll = 0.0
            req.pitch = 0.0
            req.yaw = 0.0
            req.cur_joint1 = float(self.current_joints[0])
            req.cur_joint2 = float(self.current_joints[1])
            req.cur_joint3 = float(self.current_joints[2])
            req.cur_joint4 = float(self.current_joints[3])
            req.cur_joint5 = float(self.current_joints[4])
            req.cur_joint6 = float(self.current_joints[5])
            req.kin_name = self.kin_name_fk

            self.get_logger().info(
                f"FK_CALL: joints={[round(j, 1) for j in self.current_joints]}"
            )

            future = self.kin_client.call_async(req)
            done = self._spin_wait_future(future, self.service_timeout_sec)

            if not done:
                if self.last_fk_xyz is not None and self.last_fk_rpy is not None:
                    self.get_logger().warn("FK timeout; using cached FK pose")
                    return list(self.last_fk_xyz), list(self.last_fk_rpy)
                self.get_logger().warn("FK timeout; no cached FK pose available")
                return None, None

            if future.exception() is not None:
                self.get_logger().warn(f"FK service exception: {future.exception()}")
                if self.last_fk_xyz is not None and self.last_fk_rpy is not None:
                    self.get_logger().warn("FK exception; using cached FK pose")
                    return list(self.last_fk_xyz), list(self.last_fk_rpy)
                return None, None

            res = future.result()
            if res is None:
                self.get_logger().warn("FK service returned None")
                if self.last_fk_xyz is not None and self.last_fk_rpy is not None:
                    return list(self.last_fk_xyz), list(self.last_fk_rpy)
                return None, None

            pose_xyz = [float(res.x), float(res.y), float(res.z)]
            pose_rpy = [float(res.roll), float(res.pitch), float(res.yaw)]

            self.last_fk_xyz = list(pose_xyz)
            self.last_fk_rpy = list(pose_rpy)

            self.get_logger().info(
                f"FK_RESULT: x={pose_xyz[0]:.4f}, y={pose_xyz[1]:.4f}, z={pose_xyz[2]:.4f}, "
                f"roll={pose_rpy[0]:.4f}, pitch={pose_rpy[1]:.4f}, yaw={pose_rpy[2]:.4f}"
            )
            return pose_xyz, pose_rpy
        finally:
            self.fk_busy = False

    def solve_ik_blocking(self, tar_x, tar_y, tar_z, roll, pitch, yaw):
        now = time.time()

        if self.ik_busy:
            self.get_logger().warn("IK skipped: previous IK still running")
            return None

        if (now - self.last_ik_attempt_time) < self.ik_min_interval_sec:
            self.get_logger().warn("IK skipped: rate-limited")
            return None

        if not self._service_ready("IK", timeout_sec=0.25):
            return None

        self.last_ik_attempt_time = now
        self.ik_busy = True
        try:
            req = ArmKinemarics.Request()
            req.tar_x = float(tar_x)
            req.tar_y = float(tar_y)
            req.tar_z = float(tar_z)
            req.roll = float(roll)
            req.pitch = float(pitch)
            req.yaw = float(yaw)
            req.cur_joint1 = float(self.current_joints[0])
            req.cur_joint2 = float(self.current_joints[1])
            req.cur_joint3 = float(self.current_joints[2])
            req.cur_joint4 = float(self.current_joints[3])
            req.cur_joint5 = float(self.current_joints[4])
            req.cur_joint6 = float(self.current_joints[5])
            req.kin_name = self.kin_name_ik

            self.get_logger().info(
                f"IK_CALL: x={tar_x:.4f}, y={tar_y:.4f}, z={tar_z:.4f}, "
                f"roll={roll:.6f}, pitch={pitch:.6f}, yaw={yaw:.6f}"
            )

            future = self.kin_client.call_async(req)
            done = self._spin_wait_future(future, self.service_timeout_sec)

            if not done:
                self.get_logger().warn("IK service timeout")
                return None

            if future.exception() is not None:
                self.get_logger().warn(f"IK service exception: {future.exception()}")
                return None

            res = future.result()
            if res is None:
                self.get_logger().warn("IK service returned None")
                return None

            joints = [
                float(res.joint1),
                float(res.joint2),
                float(res.joint3),
                float(res.joint4),
                float(res.joint5),
                float(res.joint6),
            ]

            if self.stabilize_wrist_joint and 0 <= self.wrist_joint_index < 6:
                joints[self.wrist_joint_index] = self.desired_wrist_deg

            if self.gripper_locked_closed:
                joints[5] = self.close_joint6_value

            self.get_logger().info(f"IK_SOLUTION: {['%.1f' % j for j in joints]}")
            return joints
        finally:
            self.ik_busy = False

    def compute_world_target(self, u, v, depth_m):
        camera_location = self.pixel_to_camera_depth((u, v), depth_m)
        pose_end_mat = np.matmul(
            self.EndToCamMat,
            self.xyz_euler_to_mat(camera_location, (0.0, 0.0, 0.0))
        )

        cur_pose_xyz, cur_pose_rpy = self.get_current_end_pose_fk_blocking()
        if cur_pose_xyz is None or cur_pose_rpy is None:
            self.get_logger().warn("compute_world_target: FK unavailable")
            return None

        end_point_mat = self.xyz_euler_to_mat(cur_pose_xyz, cur_pose_rpy)
        world_pose = np.matmul(end_point_mat, pose_end_mat)
        pose_t, _ = self.mat_to_xyz_euler(world_pose)
        pose_t[0] += self.x_offset
        pose_t[1] += self.y_offset
        pose_t[2] += self.z_offset

        self.get_logger().info(
            f"WORLD_TARGET: u={u:.1f} v={v:.1f} depth={depth_m:.3f} "
            f"world_xyz=({pose_t[0]:.4f}, {pose_t[1]:.4f}, {pose_t[2]:.4f})"
        )
        return pose_t

    def freeze_current_target(self):
        target = self.get_smoothed_target()
        if target is None:
            return False
        if not self.target_is_stable_for_freeze():
            return False

        u, v, stop_y, conf, depth = target
        world_xyz = self.compute_world_target(u, v, depth)
        if world_xyz is None:
            self.get_logger().warn("TARGET_FREEZE skipped: world target unavailable this cycle")
            return False

        self.frozen_world_point = np.array(world_xyz, dtype=np.float64)
        self.get_logger().info(
            f"TARGET_FROZEN: u={u:.1f} v={v:.1f} stop_y={stop_y:.1f} "
            f"depth={depth:.3f} world=({world_xyz[0]:.4f}, {world_xyz[1]:.4f}, {world_xyz[2]:.4f})"
        )
        return True

    def get_target_orientation(self, phase_name: str):

        if self.use_calibrated_touch_orientation:
            if phase_name == "pregrasp":
                rpy = (
                    self.pregrasp_roll_override,
                    self.pregrasp_pitch_override,
                    self.pregrasp_yaw_override,
                )
            elif phase_name == "grasp":
                rpy = (
                    self.grasp_roll_override,
                    self.grasp_pitch_override,
                    self.grasp_yaw_override,
                )
            elif phase_name == "touch":
                rpy = (
                    self.touch_roll_override,
                    self.touch_pitch_override,
                    self.touch_yaw_override,
                )
            else:
                rpy = (self.ik_roll, self.ik_pitch, self.ik_yaw)
            self.get_logger().info(
                f"{phase_name.upper()}_ORIENTATION_OVERRIDE: "
                f"roll={rpy[0]:.6f} pitch={rpy[1]:.6f} yaw={rpy[2]:.6f}"
            )
            return rpy
        return (self.ik_roll, self.ik_pitch, self.ik_yaw)

    def compute_phase_target(self, phase_name: str):
        if self.frozen_world_point is None:
            raise RuntimeError("No frozen world point")
        x, y, z = [float(v) for v in self.frozen_world_point]
        if phase_name == "pregrasp":
            extra_x = self.get_dynamic_phase_x_extra("pregrasp")
            return (
                x + self.pregrasp_x_offset_m + extra_x,
                y + self.pregrasp_y_offset_m,
                z + self.pregrasp_z_offset_m,
            )
        if phase_name == "grasp":
            extra_x = self.get_dynamic_phase_x_extra("grasp")
            return (
                x + self.grasp_x_offset_m + extra_x,
                y + self.grasp_y_offset_m,
                z + self.grasp_z_offset_m,
            )
        if phase_name == "touch":
            extra_x = self.get_dynamic_phase_x_extra("touch")
            tx = x + self.touch_x_offset_m + extra_x
            ty = y + self.touch_y_offset_m
            tz = z + self.touch_z_offset_m + self.touch_floor_extra_drop_m
            if self.use_floor_touch_z_override:
                tz = self.floor_touch_z + self.touch_surface_margin_m
            self.get_logger().info(
                f"TOUCH_TARGET: tx={tx:.4f} ty={ty:.4f} tz={tz:.4f} "
                f"use_floor_touch_z_override={self.use_floor_touch_z_override}"
            )
            return (tx, ty, tz)
        if phase_name == "lift":
            return (
                x + self.lift_x_offset_m,
                y + self.lift_y_offset_m,
                z + self.lift_z_offset_m,
            )
        raise ValueError(f"Unknown phase: {phase_name}")

    def timer_callback(self):
        self.publish_done(False)

        if self.state == "STARTUP_POSE":
            if not self.startup_pose_sent:
                self.publish_stop()
                startup_joints = list(self.startup_joints)
                startup_joints[5] = self.open_joint6_value
                self.publish_arm_joints(
                    startup_joints,
                    runtime_ms=self.startup_runtime_ms,
                    label="STARTUP_POSE_CMD",
                )
                self.startup_pose_sent = True
                self.state_start_time = time.time()
                self.get_logger().warn("STARTUP_POSE: commanded startup arm pose before WAIT_FOR_SCAN")
                return
            if not self.command_finished(self.startup_runtime_ms, self.startup_settle_sec):
                return
            self.publish_stop()
            if self.pending_patrol_release_after_startup:
                self.publish_patrol_override(False)
                self.pending_patrol_release_after_startup = False
                self.get_logger().warn("STARTUP_POSE complete -> IK node handed control back to patrol")
            self.state = "WAIT_FOR_SCAN"
            self.state_start_time = time.time()
            self.get_logger().info("STARTUP_POSE complete -> WAIT_FOR_SCAN")
            return

        if self.state == "WAIT_FOR_SCAN":
            if self.require_scan_found:
                self.publish_stop()
                return
            if self.target_is_fresh() and self.mask_seen_stable_count >= self.mask_start_stable_count:
                self.scan_found = True
                self.state = "APPROACH"
                self.state_start_time = time.time()
                self.get_logger().info("WAIT_FOR_SCAN bypassed: fresh stable mask target -> APPROACH")
                return
            self.publish_stop()
            return

        if self.state == "APPROACH":
            target = self.get_smoothed_target()
            if target is None:
                if self.should_recent_target_reacquire():
                    if self.state != "BACKUP_FOR_TARGET_REACQUIRE":
                        age = time.time() - self.last_target_time if self.last_target_time is not None else -1.0
                        self.get_logger().warn(
                            f"TARGET_TEMPORARILY_LOST: age={age:.2f}s -> backing up to reacquire sock view"
                        )
                    self.state = "BACKUP_FOR_TARGET_REACQUIRE"
                    self.state_start_time = time.time()
                    self.reacquire_backup_started_time = time.time()
                    self.publish_recent_target_backup()
                    return
                self.publish_stop()
                return
            u, v, stop_y, conf, depth = target
            ready = self.approach_with_trt(u, v, stop_y, depth)
            if not ready:
                return
            if not self.freeze_current_target():
                self.get_logger().info("APPROACH_READY but target not stable enough to freeze yet")
                return
            self.grasp_started = True
            try:
                x, y, z = self.compute_phase_target("pregrasp")
                roll, pitch, yaw = self.get_target_orientation("pregrasp")
                joints = self.solve_ik_blocking(x, y, z, roll, pitch, yaw)
                if joints is None:
                    self.get_logger().warn("PREGRASP IK unavailable; will retry")
                    return
                joints[5] = self.open_joint6_value
                self.last_pregrasp_joints = list(joints)
                self.publish_arm_joints(joints, runtime_ms=self.pregrasp_runtime_ms, label="PREGRASP_CMD")
                self.state = "WAIT_PREGRASP_SETTLE"
                self.state_start_time = time.time()
            except Exception as exc:
                self.get_logger().warn(f"PREGRASP failed: {exc}")
                self.state = "DONE"
                self.state_start_time = time.time()
            return

        if self.state == "BACKUP_FOR_TARGET_REACQUIRE":
            target = self.get_smoothed_target()
            if target is not None:
                self.publish_stop()
                self.get_logger().info("TARGET_REACQUIRED: sock back in view -> resuming APPROACH")
                self.state = "APPROACH"
                self.state_start_time = time.time()
                self.reacquire_backup_started_time = None
                return

            self.publish_recent_target_backup()
            return

        if self.state == "WAIT_PREGRASP_SETTLE":
            if not self.command_finished(self.pregrasp_runtime_ms, 0.20):
                return
            try:
                x, y, z = self.compute_phase_target("grasp")
                roll, pitch, yaw = self.get_target_orientation("grasp")
                joints = self.solve_ik_blocking(x, y, z, roll, pitch, yaw)
                if joints is None:
                    self.get_logger().warn("GRASP IK unavailable; will retry")
                    return
                joints[5] = self.open_joint6_value
                self.last_grasp_joints = list(joints)
                self.publish_arm_joints(joints, runtime_ms=self.grasp_runtime_ms, label="GRASP_CMD")
                self.state = "WAIT_GRASP_SETTLE"
                self.state_start_time = time.time()
            except Exception as exc:
                self.get_logger().warn(f"GRASP failed: {exc}")
                self.state = "DONE"
                self.state_start_time = time.time()
            return

        if self.state == "WAIT_GRASP_SETTLE":
            if not self.command_finished(self.grasp_runtime_ms, 0.20):
                return
            try:
                x, y, z = self.compute_phase_target("touch")
                roll, pitch, yaw = self.get_target_orientation("touch")
                joints = self.solve_ik_blocking(x, y, z, roll, pitch, yaw)
                if joints is None:
                    self.get_logger().warn("TOUCH IK unavailable; will retry")
                    return
                joints[5] = self.open_joint6_value
                self.last_touch_joints = list(joints)
                self.publish_arm_joints(joints, runtime_ms=self.touch_runtime_ms, label="TOUCH_CMD")
                self.state = "WAIT_TOUCH_SETTLE"
                self.state_start_time = time.time()
            except Exception as exc:
                self.get_logger().warn(f"TOUCH failed: {exc}")
                self.state = "DONE"
                self.state_start_time = time.time()
            return

        if self.state == "WAIT_TOUCH_SETTLE":
            # keep your latest timing improvement
            # if (time.time() - self.state_start_time) < (float(self.touch_runtime_ms) / 1000.0):
            #     return

            if self.last_touch_joints is not None:
                close_joints = list(self.last_touch_joints)
            else:
                close_joints = list(self.current_joints)

            self.publish_grasp_attempt(True)

            self.lock_gripper_closed()
            close_joints[5] = self.close_joint6_value
            self.last_closed_grasp_joints = list(close_joints)
            self.publish_arm_joints(close_joints, runtime_ms=self.gripper_runtime_ms, label="GRIPPER_CLOSE_CMD")
            time.sleep(0.10)
            self.publish_arm_joints(close_joints, runtime_ms=self.gripper_runtime_ms, label="GRIPPER_REINFORCE_CLOSE_CMD")
            self.state = "WAIT_CLOSE_HOLD"
            self.state_start_time = time.time()
            return

        if self.state == "WAIT_CLOSE_HOLD":
            if (time.time() - self.state_start_time) < self.close_hold_sec:
                return

            # Stay at the exact touch pose, closed.
            if self.last_closed_grasp_joints is not None:
                hold_joints = list(self.last_closed_grasp_joints)
            elif self.last_touch_joints is not None:
                hold_joints = list(self.last_touch_joints)
                hold_joints[5] = self.close_joint6_value
            else:
                hold_joints = list(self.current_joints)
                hold_joints[5] = self.close_joint6_value

            self.publish_arm_joints(
                hold_joints,
                runtime_ms=500,
                label="HOLD_CLOSED_AT_TOUCH_CMD"
            )

            # Continue as before, but without the immediate lift.
            if self.carry_pose_enabled:
                carry = list(self.carry_joints)
                carry[5] = self.close_joint6_value
                self.carry_reinforced = False
                self.publish_arm_joints(carry, runtime_ms=self.carry_runtime_ms, label="CARRY_POSE_CMD")
                self.state = "WAIT_CARRY_SETTLE"
                self.state_start_time = time.time()
                return

            self.publish_stop()
            self.publish_done(True)
            self.publish_grasp_attempt(False)
            self.state = "DONE"
            return

        if self.state == "WAIT_LIFT_SETTLE":
            elapsed = time.time() - self.state_start_time
            if (not self.post_lift_reinforced) and elapsed >= 0.25 and self.last_lift_joints is not None:
                self.publish_arm_joints(self.last_lift_joints, runtime_ms=500, label="POST_LIFT_REINFORCE_CLOSE_CMD")
                self.post_lift_reinforced = True
            if elapsed < self.lift_settle_sec:
                return
            if self.carry_pose_enabled:
                carry = list(self.carry_joints)
                carry[5] = self.close_joint6_value
                self.carry_reinforced = False
                self.publish_arm_joints(carry, runtime_ms=self.carry_runtime_ms, label="CARRY_POSE_CMD")
                self.state = "WAIT_CARRY_SETTLE"
                self.state_start_time = time.time()
                return
            self.publish_stop()
            self.publish_done(True)
            self.publish_grasp_attempt(False)
            self.state = "DONE"
            return

        if self.state == "WAIT_CARRY_SETTLE":
            elapsed = time.time() - self.state_start_time
            if (not self.carry_reinforced) and elapsed >= 0.45:
                carry = list(self.carry_joints)
                carry[5] = self.close_joint6_value
                self.publish_arm_joints(carry, runtime_ms=500, label="CARRY_REINFORCE_CLOSE_CMD")
                self.carry_reinforced = True
            if elapsed < self.carry_settle_sec:
                return

            # Handoff to carry_sock_to_person_node
            self.publish_stop()
            self.publish_done(True)
            self.publish_grasp_attempt(False)

            # Keep patrol paused while carry node takes over
            self.publish_patrol_override(True)

            # Trigger carry node
            self.publish_grasp_complete(True)

            self.handoff_to_carry_active = True

            self.get_logger().info(
                "Carry pose settled -> published /sock/grasp_complete = True, "
                "holding override for carry_sock_to_person_node"
            )

            self.state = "DONE"
            return

        if self.state == "DONE":
            # Stay idle after grasp handoff.
            # carry_sock_to_person_node owns /cmd_vel now.
            # We wait here until /sock/grasp_retry_requested arrives.
            # Do not continuously publish stop after handoff or it will block carry node motion.
            self.publish_done(True)
            return


def main(args=None):
    rclpy.init(args=args)
    node = GraspSockIKNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()