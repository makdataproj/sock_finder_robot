#!/usr/bin/env python3
import math
import time

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float32, Float32MultiArray
from arm_msgs.msg import ArmJoints


class CarrySockToPersonNode(Node):
    def __init__(self):
        super().__init__("carry_sock_to_person_node")

        # -----------------------------
        # Parameters
        # -----------------------------
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("patrol_override_topic", "/sock/patrol_override_active")
        self.declare_parameter("handoff_complete_topic", "/person/handoff_complete")
        self.declare_parameter("retry_request_topic", "/sock/grasp_retry_requested")
        self.declare_parameter("grasp_complete_topic", "/sock/grasp_complete")

        self.declare_parameter("sock_detector_enable_topic", "/sock/detector_enable")
        self.declare_parameter("person_detector_enable_topic", "/person/detector_enable")

        self.declare_parameter("person_target_topic", "/person/target")
        self.declare_parameter("force_topic", "/force_sensor/raw")

        # Arm reset / patrol handoff pose
        self.declare_parameter("arm_topic", "/arm6_joints")
        self.declare_parameter("post_handoff_joint1", 90.0)
        self.declare_parameter("post_handoff_joint2", 90.0)
        self.declare_parameter("post_handoff_joint3", 45.0)
        self.declare_parameter("post_handoff_joint4", 0.0)
        self.declare_parameter("post_handoff_joint5", 90.0)
        self.declare_parameter("post_handoff_joint6", 90.0)
        self.declare_parameter("post_handoff_runtime_ms", 1500)

        # Follow / alignment behavior
        self.declare_parameter("image_center_x", 320.0)
        self.declare_parameter("angular_gain", 0.0028)
        self.declare_parameter("max_angular_speed", 0.35)
        self.declare_parameter("align_deadband_px", 22.0)
        self.declare_parameter("move_while_turning_px", 85.0)

        # Search behavior
        self.declare_parameter("search_angular_speed", 0.28)
        self.declare_parameter("search_pause_sec", 0.20)
        self.declare_parameter("person_lost_timeout_sec", 0.60)
        self.declare_parameter("full_spin_timeout_sec", 14.0)

        # Approach behavior
        # NOTE:
        # width-based stopping cannot guarantee exact millimeters in the real world.
        # We stop conservatively before contact using bbox width as a proxy.
        self.declare_parameter("target_person_width_px", 190.0)
        self.declare_parameter("stop_width_margin_px", 12.0)
        self.declare_parameter("linear_gain", 0.0012)
        self.declare_parameter("min_linear_speed", 0.03)
        self.declare_parameter("max_linear_speed", 0.09)
        self.declare_parameter("min_trackable_width_px", 18.0)

        # Extra safety behavior near target
        self.declare_parameter("close_align_required_px", 12.0)
        self.declare_parameter("final_stop_hold_sec", 0.75)
        self.declare_parameter("close_confirm_required", 3)

        # FSR / removal detection
        self.declare_parameter("fsr_ready_required", 3)
        self.declare_parameter("fsr_drop_threshold", 1200.0)
        self.declare_parameter("fsr_required_consecutive", 4)

        # Debug
        self.declare_parameter("debug_period_sec", 0.5)

        gp = lambda name: self.get_parameter(name).value

        self.cmd_vel_topic = str(gp("cmd_vel_topic"))
        self.patrol_override_topic = str(gp("patrol_override_topic"))
        self.handoff_complete_topic = str(gp("handoff_complete_topic"))
        self.retry_request_topic = str(gp("retry_request_topic"))
        self.grasp_complete_topic = str(gp("grasp_complete_topic"))
        self.sock_detector_enable_topic = str(gp("sock_detector_enable_topic"))
        self.person_detector_enable_topic = str(gp("person_detector_enable_topic"))
        self.person_target_topic = str(gp("person_target_topic"))
        self.force_topic = str(gp("force_topic"))

        self.arm_topic = str(gp("arm_topic"))
        self.post_handoff_joints = [
            float(gp("post_handoff_joint1")),
            float(gp("post_handoff_joint2")),
            float(gp("post_handoff_joint3")),
            float(gp("post_handoff_joint4")),
            float(gp("post_handoff_joint5")),
            float(gp("post_handoff_joint6")),
        ]
        self.post_handoff_runtime_ms = int(gp("post_handoff_runtime_ms"))

        self.image_center_x = float(gp("image_center_x"))
        self.angular_gain = float(gp("angular_gain"))
        self.max_angular_speed = float(gp("max_angular_speed"))
        self.align_deadband_px = float(gp("align_deadband_px"))
        self.move_while_turning_px = float(gp("move_while_turning_px"))

        self.search_angular_speed = float(gp("search_angular_speed"))
        self.search_pause_sec = float(gp("search_pause_sec"))
        self.person_lost_timeout_sec = float(gp("person_lost_timeout_sec"))
        self.full_spin_timeout_sec = float(gp("full_spin_timeout_sec"))

        self.target_person_width_px = float(gp("target_person_width_px"))
        self.stop_width_margin_px = float(gp("stop_width_margin_px"))
        self.linear_gain = float(gp("linear_gain"))
        self.min_linear_speed = float(gp("min_linear_speed"))
        self.max_linear_speed = float(gp("max_linear_speed"))
        self.min_trackable_width_px = float(gp("min_trackable_width_px"))

        self.close_align_required_px = float(gp("close_align_required_px"))
        self.final_stop_hold_sec = float(gp("final_stop_hold_sec"))
        self.close_confirm_required = int(gp("close_confirm_required"))

        self.fsr_ready_required = int(gp("fsr_ready_required"))
        self.fsr_drop_threshold = float(gp("fsr_drop_threshold"))
        self.fsr_required_consecutive = int(gp("fsr_required_consecutive"))

        self.debug_period_sec = float(gp("debug_period_sec"))

        # -----------------------------
        # Publishers
        # -----------------------------
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.arm_pub = self.create_publisher(ArmJoints, self.arm_topic, 10)
        self.override_pub = self.create_publisher(Bool, self.patrol_override_topic, 10)
        self.handoff_pub = self.create_publisher(Bool, self.handoff_complete_topic, 10)
        self.retry_pub = self.create_publisher(Bool, self.retry_request_topic, 10)
        self.sock_detector_enable_pub = self.create_publisher(Bool, self.sock_detector_enable_topic, 10)
        self.person_detector_enable_pub = self.create_publisher(Bool, self.person_detector_enable_topic, 10)
        self.grasp_complete_pub = self.create_publisher(Bool, self.grasp_complete_topic, 10)

        # -----------------------------
        # Subscriptions
        # -----------------------------
        self.create_subscription(Float32MultiArray, self.person_target_topic, self.person_cb, 10)
        self.create_subscription(Float32, self.force_topic, self.fsr_cb, 10)
        self.create_subscription(Bool, self.grasp_complete_topic, self.grasp_cb, 10)

        # -----------------------------
        # State
        # -----------------------------
        self.active = False
        self.person = None
        self.person_last_seen_time = 0.0

        self.fsr_val = None
        self.fsr_ready_count = 0
        self.low_count = 0

        self.last_debug_time = 0.0

        # Search / approach states
        self.mode = "idle"  # idle, searching, tracking, reached_person
        self.search_start_time = 0.0
        self.search_pause_until = 0.0
        self.reached_person_time = None
        self.close_confirm_count = 0

        self.timer = self.create_timer(0.05, self.loop)

        # Startup default: sock detector on, person detector off
        self.publish_detector_enable(sock_enabled=True, person_enabled=False)

        self.get_logger().info("CarrySockToPersonNode ready")
        self.get_logger().info(
            f"Using SERIAL FSR topic: {self.force_topic}, person topic: {self.person_target_topic}"
        )
        self.get_logger().info("Startup mode: sock detector ON, person detector OFF")

    # -------------------------------------------------
    # Helpers
    # -------------------------------------------------
    def publish_detector_enable(self, sock_enabled: bool, person_enabled: bool) -> None:
        self.sock_detector_enable_pub.publish(Bool(data=bool(sock_enabled)))
        self.person_detector_enable_pub.publish(Bool(data=bool(person_enabled)))

    def stop_robot(self) -> None:
        self.cmd_pub.publish(Twist())

    def publish_post_handoff_pose(self) -> None:
        msg = ArmJoints()
        msg.joint1 = int(round(self.post_handoff_joints[0]))
        msg.joint2 = int(round(self.post_handoff_joints[1]))
        msg.joint3 = int(round(self.post_handoff_joints[2]))
        msg.joint4 = int(round(self.post_handoff_joints[3]))
        msg.joint5 = int(round(self.post_handoff_joints[4]))
        msg.joint6 = int(round(self.post_handoff_joints[5]))
        msg.time = int(self.post_handoff_runtime_ms)
        self.arm_pub.publish(msg)
        self.get_logger().warn(
            f"POST_HANDOFF_POSE_CMD: "
            f"{[msg.joint1, msg.joint2, msg.joint3, msg.joint4, msg.joint5, msg.joint6]} "
            f"time={msg.time}"
        )

    def maybe_debug_log(self, text: str) -> None:
        now = time.time()
        if (now - self.last_debug_time) >= self.debug_period_sec:
            self.get_logger().info(text)
            self.last_debug_time = now

    def restart_sock_pipeline(self) -> None:
        self.stop_robot()

        # Move arm back to the known patrol/startup pose as part of handoff.
        # This is the programmatic equivalent of:
        # ros2 topic pub /arm6_joints arm_msgs/msg/ArmJoints
        # "{joint1: 90, joint2: 90, joint3: 45, joint4: 0, joint5: 90, joint6: 90, time: 1500}" --once
        self.publish_post_handoff_pose()

        # Person handoff/removal complete
        self.handoff_pub.publish(Bool(data=True))

        # Give control back to patrol
        self.override_pub.publish(Bool(data=False))

        # Switch detectors back
        self.publish_detector_enable(sock_enabled=True, person_enabled=False)

        # Clear grasp-complete latch for next cycle
        self.grasp_complete_pub.publish(Bool(data=False))

        # Ask IK node to reset / start again after patrol handoff
        self.retry_pub.publish(Bool(data=True))

        self.active = False
        self.person = None
        self.person_last_seen_time = 0.0
        self.fsr_val = None
        self.fsr_ready_count = 0
        self.low_count = 0
        self.mode = "idle"
        self.search_start_time = 0.0
        self.search_pause_until = 0.0
        self.reached_person_time = None
        self.close_confirm_count = 0

        self.get_logger().warn(
            "Sock removed -> post-handoff arm pose published, patrol override FALSE, "
            "person detector OFF, sock detector ON, IK retry requested"
        )

    def begin_search(self) -> None:
        self.mode = "searching"
        self.search_start_time = time.time()
        self.search_pause_until = 0.0
        self.reached_person_time = None
        self.close_confirm_count = 0
        self.stop_robot()
        self.get_logger().info("SEARCHING: spinning to find person")

    def valid_person(self) -> bool:
        if self.person is None or len(self.person) < 3:
            return False
        try:
            width = float(self.person[2])
        except Exception:
            return False
        return width >= self.min_trackable_width_px

    # -------------------------------------------------
    # Callbacks
    # -------------------------------------------------
    def grasp_cb(self, msg: Bool) -> None:
        if msg.data:
            self.active = True
            self.person = None
            self.person_last_seen_time = 0.0
            self.fsr_val = None
            self.fsr_ready_count = 0
            self.low_count = 0
            self.close_confirm_count = 0

            # Carry mode owns the base
            self.override_pub.publish(Bool(data=True))
            self.publish_detector_enable(sock_enabled=False, person_enabled=True)

            self.begin_search()
            self.get_logger().info("Carry mode active: sock detector OFF, person detector ON")
        else:
            self.active = False
            self.mode = "idle"
            self.stop_robot()
            self.publish_detector_enable(sock_enabled=True, person_enabled=False)
            self.get_logger().info("Carry mode inactive: sock detector ON, person detector OFF")

    def person_cb(self, msg: Float32MultiArray) -> None:
        data = list(msg.data)
        self.person = data
        self.person_last_seen_time = time.time()

        if self.active and self.valid_person() and self.mode in ("searching", "tracking"):
            if self.mode != "tracking":
                self.get_logger().info("PERSON FOUND: switching from searching to tracking")
            self.mode = "tracking"

    def fsr_cb(self, msg: Float32) -> None:
        self.fsr_val = float(msg.data)
        self.fsr_ready_count += 1

        if not self.active:
            return

        if self.fsr_ready_count < self.fsr_ready_required:
            return

        if self.fsr_val < self.fsr_drop_threshold:
            self.low_count += 1
        else:
            self.low_count = 0

        if self.low_count >= self.fsr_required_consecutive:
            self.get_logger().warn(
                f"FSR drop detected: force={self.fsr_val:.1f}, low_count={self.low_count}"
            )
            self.restart_sock_pipeline()

    # -------------------------------------------------
    # Main loop
    # -------------------------------------------------
    def loop(self) -> None:
        if not self.active:
            return

        # Keep patrol paused while carrying
        self.override_pub.publish(Bool(data=True))

        # Wait for valid FSR stream
        if self.fsr_ready_count < self.fsr_ready_required or self.fsr_val is None:
            self.stop_robot()
            self.maybe_debug_log(
                f"Waiting for FSR... ready_count={self.fsr_ready_count}/{self.fsr_ready_required}, "
                f"force={self.fsr_val}"
            )
            return

        now = time.time()

        if self.mode == "searching":
            if self.valid_person() and (now - self.person_last_seen_time) <= self.person_lost_timeout_sec:
                self.mode = "tracking"
                return

            if now < self.search_pause_until:
                self.stop_robot()
                return

            twist = Twist()
            twist.angular.z = self.search_angular_speed
            self.cmd_pub.publish(twist)

            elapsed = now - self.search_start_time
            self.maybe_debug_log(
                f"SEARCHING: spinning 360 for person, elapsed={elapsed:.2f}s, force={self.fsr_val:.1f}"
            )

            if elapsed >= self.full_spin_timeout_sec:
                self.get_logger().warn("Completed full spin without a person. Continuing search...")
                self.search_start_time = now
                self.search_pause_until = now + self.search_pause_sec
                self.stop_robot()
            return

        if self.mode == "reached_person":
            # Do not stay stuck here forever. If the person target is lost or no longer valid,
            # go back to 360 search immediately.
            if not self.valid_person() or (now - self.person_last_seen_time) > self.person_lost_timeout_sec:
                self.get_logger().warn("REACHED_PERSON but person target lost -> resuming 360 search")
                self.begin_search()
                return

            self.stop_robot()
            if self.reached_person_time is None:
                self.reached_person_time = now

            self.maybe_debug_log(
                f"REACHED_PERSON: holding position safely, force={self.fsr_val:.1f}, "
                f"hold_elapsed={now - self.reached_person_time:.2f}s"
            )
            return

        # Tracking mode
        if not self.valid_person() or (now - self.person_last_seen_time) > self.person_lost_timeout_sec:
            self.get_logger().warn("Lost person target -> resuming 360 search")
            self.begin_search()
            return

        # Expected target formats:
        # [cx, cy, w, h, conf]
        # [cx, cy, w, conf]
        cx = float(self.person[0])
        width = float(self.person[2])

        dx = cx - self.image_center_x
        err = self.target_person_width_px - width

        twist = Twist()

        # Angular correction
        if abs(dx) > self.align_deadband_px:
            twist.angular.z = max(
                min(-dx * self.angular_gain, self.max_angular_speed),
                -self.max_angular_speed
            )
        else:
            twist.angular.z = 0.0

        # Stop rule before contact.
        # Require a few consecutive close-enough samples so we do not latch into
        # "holding position safely" off a brief bad detection.
        close_enough = width >= (self.target_person_width_px - self.stop_width_margin_px)
        aligned_close = abs(dx) <= self.close_align_required_px

        if close_enough and aligned_close:
            self.close_confirm_count += 1
        else:
            self.close_confirm_count = 0

        if self.close_confirm_count >= self.close_confirm_required:
            self.stop_robot()
            if self.mode != "reached_person":
                self.mode = "reached_person"
                self.reached_person_time = now
                self.get_logger().warn(
                    f"Reached safe handoff distance. STOPPING. width={width:.1f}, dx={dx:.1f}, "
                    f"confirm_count={self.close_confirm_count}"
                )
            return

        # Move toward the person once locked.
        # If very off-center, rotate only.
        # If moderately aligned, rotate and creep forward at the same time.
        if abs(dx) <= self.move_while_turning_px and err > self.stop_width_margin_px:
            speed = min(self.linear_gain * err, self.max_linear_speed)
            if speed < self.min_linear_speed:
                speed = self.min_linear_speed
            twist.linear.x = speed
        else:
            twist.linear.x = 0.0

        self.cmd_pub.publish(twist)

        self.maybe_debug_log(
            f"TRACKING: force={self.fsr_val:.1f}, mode={self.mode}, "
            f"cx={cx:.1f}, width={width:.1f}, dx={dx:.1f}, err={err:.1f}, "
            f"linear={twist.linear.x:.3f}, angular={twist.angular.z:.3f}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = CarrySockToPersonNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
