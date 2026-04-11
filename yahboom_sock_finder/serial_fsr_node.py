#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import time
from collections import deque

import serial
import serial.serialutil

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float32, Float32MultiArray, String
from arm_msgs.msg import ArmJoints


class FSRGraspRecoveryNode(Node):
    def __init__(self):
        super().__init__("fsr_grasp_recovery_node")

        # -----------------------------
        # Serial / FSR parameters
        # -----------------------------
        self.declare_parameter("port", "/dev/esp32_fsr")
        self.declare_parameter("baudrate", 115200)
        self.declare_parameter("read_timeout", 0.1)
        self.declare_parameter("reconnect_sec", 2.0)

        # Topics
        self.declare_parameter("grasp_attempt_topic", "/sock/grasp_attempt")
        self.declare_parameter("retry_request_topic", "/sock/grasp_retry_requested")
        self.declare_parameter("bbox_topic", "/sock/best_bbox")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("arm_topic", "/arm6_joints")

        # Debug publish topics
        self.declare_parameter("publish_force_topic", "/gripper/force")
        self.declare_parameter("publish_raw_topic", "/gripper/raw")
        self.declare_parameter("state_topic", "/sock/fsr_recovery_state")

        # Simple success rule
        self.declare_parameter("success_threshold", 1800.0)

        # Retry / backup behavior
        self.declare_parameter("enable_retry_on_fail", True)
        self.declare_parameter("backup_speed", -0.035)
        self.declare_parameter("bbox_timeout_sec", 0.6)
        self.declare_parameter("bbox_min_confidence", 0.35)
        self.declare_parameter("retry_pulse_sec", 0.20)

        # Restart pose
        self.declare_parameter("restart_joint1", 90.0)
        self.declare_parameter("restart_joint2", 90.0)
        self.declare_parameter("restart_joint3", 45.0)
        self.declare_parameter("restart_joint4", 0.0)
        self.declare_parameter("restart_joint5", 90.0)
        self.declare_parameter("restart_joint6", 90.0)
        self.declare_parameter("restart_runtime_ms", 1500)

        # Debug
        self.declare_parameter("debug", True)
        self.declare_parameter("log_every_n_samples", 1)

        gp = lambda name: self.get_parameter(name).value

        self.port = str(gp("port"))
        self.baudrate = int(gp("baudrate"))
        self.read_timeout = float(gp("read_timeout"))
        self.reconnect_sec = float(gp("reconnect_sec"))

        self.grasp_attempt_topic = str(gp("grasp_attempt_topic"))
        self.retry_request_topic = str(gp("retry_request_topic"))
        self.bbox_topic = str(gp("bbox_topic"))
        self.cmd_vel_topic = str(gp("cmd_vel_topic"))
        self.arm_topic = str(gp("arm_topic"))

        self.publish_force_topic = str(gp("publish_force_topic"))
        self.publish_raw_topic = str(gp("publish_raw_topic"))
        self.state_topic = str(gp("state_topic"))

        self.success_threshold = float(gp("success_threshold"))

        self.enable_retry_on_fail = bool(gp("enable_retry_on_fail"))
        self.backup_speed = float(gp("backup_speed"))
        self.bbox_timeout_sec = float(gp("bbox_timeout_sec"))
        self.bbox_min_confidence = float(gp("bbox_min_confidence"))
        self.retry_pulse_sec = float(gp("retry_pulse_sec"))

        self.restart_joints = [
            float(gp("restart_joint1")),
            float(gp("restart_joint2")),
            float(gp("restart_joint3")),
            float(gp("restart_joint4")),
            float(gp("restart_joint5")),
            float(gp("restart_joint6")),
        ]
        self.restart_runtime_ms = int(gp("restart_runtime_ms"))

        self.debug = bool(gp("debug"))
        self.log_every_n_samples = max(1, int(gp("log_every_n_samples")))

        # -----------------------------
        # Publishers
        # -----------------------------
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.arm_pub = self.create_publisher(ArmJoints, self.arm_topic, 10)

        self.success_pub = self.create_publisher(Bool, "/sock/grasp_fsr_success", 10)
        self.fail_pub = self.create_publisher(Bool, "/sock/grasp_fsr_fail", 10)
        self.retry_pub = self.create_publisher(Bool, self.retry_request_topic, 10)
        self.state_pub = self.create_publisher(String, self.state_topic, 10)

        self.force_pub = self.create_publisher(Float32, self.publish_force_topic, 10)
        self.raw_pub = self.create_publisher(String, self.publish_raw_topic, 10)

        # -----------------------------
        # Subscribers
        # -----------------------------
        self.create_subscription(Bool, self.grasp_attempt_topic, self.grasp_attempt_callback, 10)
        self.create_subscription(Float32MultiArray, self.bbox_topic, self.bbox_callback, 10)

        # -----------------------------
        # Internal state
        # -----------------------------
        self.ser = None

        self.current_force = 0.0
        self.last_line = ""
        self.last_force_time = None
        self.sample_count = 0
        self.force_hist = deque(maxlen=3)

        self.monitoring = False
        self.last_grasp_start_time = None

        self.latest_bbox = None
        self.last_bbox_time = None

        self.state = "IDLE"
        self.retry_pulse_active = False
        self.retry_pulse_end_time = None
        self.restart_pose_sent = False

        # -----------------------------
        # Timers
        # -----------------------------
        self.read_timer = self.create_timer(0.02, self.read_serial)
        self.reconnect_timer = self.create_timer(self.reconnect_sec, self.try_connect)
        self.control_timer = self.create_timer(0.05, self.control_loop)

        self.try_connect()

        # -----------------------------
        # Startup fingerprint
        # -----------------------------
        self.get_logger().warn("============================================================")
        self.get_logger().warn("=== FSR DIRECT SERIAL RETRY VERSION V2 ===")
        self.get_logger().warn("============================================================")
        self.get_logger().info(
            f"port={self.port}, baudrate={self.baudrate}, grasp_attempt_topic={self.grasp_attempt_topic}"
        )
        self.get_logger().info(
            f"success_threshold={self.success_threshold}, bbox_topic={self.bbox_topic}, "
            f"bbox_min_confidence={self.bbox_min_confidence}"
        )
        self.get_logger().warn("IMPORTANT: do NOT run arduino-cli monitor while this node is running.")

    # -------------------------------------------------
    # Helpers
    # -------------------------------------------------
    def set_state(self, state: str):
        if self.state != state:
            self.get_logger().info(f"STATE: {self.state} -> {state}")
            self.state = state
            msg = String()
            msg.data = state
            self.state_pub.publish(msg)

    def publish_bool(self, pub, value: bool):
        msg = Bool()
        msg.data = bool(value)
        pub.publish(msg)

    def publish_stop(self):
        self.cmd_pub.publish(Twist())

    def publish_backup(self):
        cmd = Twist()
        cmd.linear.x = float(self.backup_speed)
        self.cmd_pub.publish(cmd)

    def publish_restart_pose(self):
        msg = ArmJoints()
        msg.joint1 = int(round(self.restart_joints[0]))
        msg.joint2 = int(round(self.restart_joints[1]))
        msg.joint3 = int(round(self.restart_joints[2]))
        msg.joint4 = int(round(self.restart_joints[3]))
        msg.joint5 = int(round(self.restart_joints[4]))
        msg.joint6 = int(round(self.restart_joints[5]))
        msg.time = int(self.restart_runtime_ms)
        self.arm_pub.publish(msg)
        self.get_logger().warn(
            f"RESTART_POSE_CMD: {[msg.joint1, msg.joint2, msg.joint3, msg.joint4, msg.joint5, msg.joint6]} "
            f"runtime_ms={msg.time}"
        )

    def close_serial(self):
        try:
            if self.ser is not None:
                self.ser.close()
        except Exception:
            pass
        self.ser = None

    def try_connect(self):
        if self.ser is not None and self.ser.is_open:
            return
        try:
            self.get_logger().info(f"[SERIAL] Connecting to {self.port} @ {self.baudrate}")
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.read_timeout)
            self.get_logger().warn("[SERIAL] Connected successfully")
        except Exception as e:
            self.ser = None
            self.get_logger().warn(f"[SERIAL] Connect failed: {e}")

    def parse_force_value(self, decoded: str):
        text = decoded.strip()
        if not text:
            return None

        try:
            return float(text)
        except ValueError:
            pass

        m = re.search(r"Raw:\s*([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
        if m:
            return float(m.group(1))

        return None

    def bbox_is_fresh(self):
        if self.last_bbox_time is None or self.latest_bbox is None:
            return False
        if (time.time() - self.last_bbox_time) > self.bbox_timeout_sec:
            return False
        conf = float(self.latest_bbox[4])
        return conf >= self.bbox_min_confidence

    # -------------------------------------------------
    # Serial read
    # -------------------------------------------------
    def read_serial(self):
        if self.ser is None or not self.ser.is_open:
            return

        try:
            line = self.ser.readline()
            if not line:
                return

            decoded = line.decode("utf-8", errors="ignore").strip()
            if not decoded:
                return

            self.last_line = decoded

            raw_msg = String()
            raw_msg.data = decoded
            self.raw_pub.publish(raw_msg)

            value = self.parse_force_value(decoded)
            if value is None:
                if self.debug:
                    self.get_logger().warn(f"[SERIAL PARSE] Could not parse: '{decoded}'")
                return

            self.current_force = float(value)
            self.last_force_time = time.time()
            self.sample_count += 1

            self.force_hist.append(self.current_force)
            filtered = sum(self.force_hist) / len(self.force_hist)

            force_msg = Float32()
            force_msg.data = self.current_force
            self.force_pub.publish(force_msg)

            if self.debug and (self.sample_count % self.log_every_n_samples == 0):
                mode = "GRASP_ACTIVE" if self.monitoring else self.state
                self.get_logger().info(
                    f"FSR_STREAM [{mode}] raw={self.current_force:.1f} filtered={filtered:.1f} "
                    f"line='{decoded}'"
                )

        except Exception as e:
            self.get_logger().warn(f"[SERIAL] Read error: {e}")
            self.close_serial()

    # -------------------------------------------------
    # Callbacks
    # -------------------------------------------------
    def bbox_callback(self, msg: Float32MultiArray):
        if len(msg.data) < 5:
            return
        self.latest_bbox = [float(v) for v in msg.data[:5]]
        self.last_bbox_time = time.time()

    def grasp_attempt_callback(self, msg: Bool):
        active = bool(msg.data)

        if active:
            self.monitoring = True
            self.last_grasp_start_time = time.time()
            self.publish_bool(self.success_pub, False)
            self.publish_bool(self.fail_pub, False)
            self.get_logger().warn(
                f"GRASP_STARTED: latest_raw={self.current_force:.1f} last_line='{self.last_line}'"
            )
            return

        if not self.monitoring:
            return

        self.monitoring = False

        elapsed = 0.0
        if self.last_grasp_start_time is not None:
            elapsed = time.time() - self.last_grasp_start_time

        final_raw = self.current_force

        self.get_logger().warn(
            f"FINAL_FORCE_CHECK: raw={final_raw:.1f} threshold={self.success_threshold:.1f} "
            f"elapsed={elapsed:.2f}s last_line='{self.last_line}'"
        )

        if final_raw >= self.success_threshold:
            self.get_logger().warn("GRASP SUCCESS (direct serial final threshold met)")
            self.publish_bool(self.success_pub, True)
            self.publish_bool(self.fail_pub, False)
            self.publish_stop()
            self.set_state("IDLE")
            return

        self.get_logger().warn("GRASP FAIL -> restart pose, backup, reacquire bbox, retry")
        self.publish_bool(self.success_pub, False)
        self.publish_bool(self.fail_pub, True)

        if self.enable_retry_on_fail:
            self.restart_pose_sent = False
            self.set_state("BACKUP_UNTIL_BBOX")

    # -------------------------------------------------
    # Control loop
    # -------------------------------------------------
    def control_loop(self):
        if self.state == "BACKUP_UNTIL_BBOX":
            if not self.restart_pose_sent:
                self.publish_restart_pose()
                self.restart_pose_sent = True

            if self.bbox_is_fresh():
                self.publish_stop()
                self.publish_bool(self.retry_pub, True)
                self.retry_pulse_active = True
                self.retry_pulse_end_time = time.time() + self.retry_pulse_sec
                self.get_logger().warn(
                    f"BBOX_REACQUIRED: conf={self.latest_bbox[4]:.2f} -> RETRY_REQUESTED"
                )
                self.set_state("WAIT_RETRY_PULSE_END")
                return

            self.publish_backup()
            return

        if self.state == "WAIT_RETRY_PULSE_END":
            self.publish_stop()
            if self.retry_pulse_active and time.time() >= self.retry_pulse_end_time:
                self.publish_bool(self.retry_pub, False)
                self.retry_pulse_active = False
                self.retry_pulse_end_time = None
                self.set_state("IDLE")
            return

        if self.state == "IDLE":
            return

    # -------------------------------------------------
    # Cleanup
    # -------------------------------------------------
    def destroy_node(self):
        try:
            self.publish_stop()
        except Exception:
            pass
        self.close_serial()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = FSRGraspRecoveryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
