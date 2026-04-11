#!/usr/bin/env python3
import math
import time
from typing import List, Dict, Optional

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, Twist
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from std_msgs.msg import Bool
from tf2_ros import Buffer, TransformListener


def yaw_to_quaternion(z_yaw: float):
    from geometry_msgs.msg import Quaternion
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(z_yaw / 2.0)
    q.w = math.cos(z_yaw / 2.0)
    return q


class PatrolNode(Node):
    def __init__(self):
        super().__init__("patrol_node")

        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_footprint")
        self.declare_parameter("loop_forever", True)
        self.declare_parameter("ping_pong", False)
        self.declare_parameter("pause_at_waypoint_sec", 2.0)
        self.declare_parameter("startup_delay_sec", 2.0)
        self.declare_parameter("control_period_sec", 0.5)
        self.declare_parameter("goal_timeout_sec", 20.0)
        self.declare_parameter("min_waypoint_separation_m", 0.25)
        self.declare_parameter("ignore_goal_yaw", True)

        self.declare_parameter("patrol_override_topic", "/sock/patrol_override_active")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")

        self.declare_parameter("waypoints", [
            1.971, -1.198, -2.648,
            1.619, -1.386, -2.648,
            1.971, -1.198,  0.494,
            2.323, -1.010,  0.494,
        ])

        self.map_frame = str(self.get_parameter("map_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.loop_forever = bool(self.get_parameter("loop_forever").value)
        self.ping_pong = bool(self.get_parameter("ping_pong").value)
        self.pause_at_waypoint_sec = float(self.get_parameter("pause_at_waypoint_sec").value)
        self.startup_delay_sec = float(self.get_parameter("startup_delay_sec").value)
        self.control_period_sec = float(self.get_parameter("control_period_sec").value)
        self.goal_timeout_sec = float(self.get_parameter("goal_timeout_sec").value)
        self.min_waypoint_separation_m = float(self.get_parameter("min_waypoint_separation_m").value)
        self.ignore_goal_yaw = bool(self.get_parameter("ignore_goal_yaw").value)
        self.patrol_override_topic = str(self.get_parameter("patrol_override_topic").value)
        self.cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)

        raw_waypoints = self.get_parameter("waypoints").value
        self.waypoints = self._parse_waypoints(raw_waypoints)
        if not self.waypoints:
            raise ValueError("No valid patrol waypoints were provided.")

        self.navigator = BasicNavigator()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.create_subscription(
            Bool,
            self.patrol_override_topic,
            self._patrol_override_callback,
            10
        )

        self.current_index = 0
        self.direction = 1

        self.nav_ready = False
        self.goal_sent = False
        self.goal_start_time: Optional[float] = None
        self.start_time = time.time()
        self.pause_until: Optional[float] = None
        self.finished = False

        self.patrol_override_active = False
        self.was_paused_by_override = False
        self.nav_goal_active = False

        self.timer = self.create_timer(self.control_period_sec, self._tick)

        self.get_logger().info("==================================================")
        self.get_logger().info("PATROL NODE STARTED")
        self.get_logger().info(f"Loaded {len(self.waypoints)} waypoint(s)")
        for i, wp in enumerate(self.waypoints):
            self.get_logger().info(
                f"WAYPOINT[{i}] x={wp['x']:.3f}, y={wp['y']:.3f}, yaw={wp['yaw']:.3f}"
            )
        self.get_logger().info(
            f"loop_forever={self.loop_forever}, ping_pong={self.ping_pong}, "
            f"ignore_goal_yaw={self.ignore_goal_yaw}"
        )
        self.get_logger().info(
            f"goal_timeout_sec={self.goal_timeout_sec:.1f}, "
            f"min_waypoint_separation_m={self.min_waypoint_separation_m:.2f}"
        )
        self.get_logger().info(
            f"patrol_override_topic={self.patrol_override_topic}, cmd_vel_topic={self.cmd_vel_topic}"
        )
        self.get_logger().info("==================================================")

    def _publish_stop(self):
        self.cmd_pub.publish(Twist())

    def _cancel_nav_if_active(self):
        try:
            self.navigator.cancelTask()
        except Exception as exc:
            self.get_logger().warn(f"Failed to cancel Nav2 task cleanly: {exc}")
        self.nav_goal_active = False
        self.goal_sent = False
        self.goal_start_time = None

    def _patrol_override_callback(self, msg: Bool):
        new_value = bool(msg.data)

        if new_value == self.patrol_override_active:
            return

        self.patrol_override_active = new_value

        if self.patrol_override_active:
            self.get_logger().warn("PATROL_OVERRIDE: IK node took control -> canceling patrol/nav2")
            self.was_paused_by_override = True
            self._cancel_nav_if_active()
            self._publish_stop()   # one-time stop is okay here
        else:
            self.get_logger().warn("PATROL_OVERRIDE cleared -> patrol may resume current waypoint")

    def _parse_waypoints(self, raw_waypoints) -> List[Dict[str, float]]:
        parsed: List[Dict[str, float]] = []

        if not isinstance(raw_waypoints, (list, tuple)):
            self.get_logger().error("waypoints parameter is not a list")
            return parsed

        if len(raw_waypoints) % 3 != 0:
            self.get_logger().error(
                f"waypoints length must be a multiple of 3, got {len(raw_waypoints)}"
            )
            return parsed

        for i in range(0, len(raw_waypoints), 3):
            try:
                parsed.append({
                    "x": float(raw_waypoints[i]),
                    "y": float(raw_waypoints[i + 1]),
                    "yaw": float(raw_waypoints[i + 2]),
                })
            except Exception as exc:
                self.get_logger().warn(
                    f"Failed to parse waypoint triple starting at index {i}: {exc}"
                )
        return parsed

    def _make_pose(self, x: float, y: float, yaw: float) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = self.map_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0

        final_yaw = 0.0 if self.ignore_goal_yaw else yaw
        pose.pose.orientation = yaw_to_quaternion(final_yaw)
        return pose

    def _get_current_pose(self) -> Optional[Dict[str, float]]:
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                rclpy.time.Time()
            )
            x = tf.transform.translation.x
            y = tf.transform.translation.y

            qx = tf.transform.rotation.x
            qy = tf.transform.rotation.y
            qz = tf.transform.rotation.z
            qw = tf.transform.rotation.w

            yaw = math.atan2(
                2.0 * (qw * qz + qx * qy),
                1.0 - 2.0 * (qy * qy + qz * qz)
            )
            return {"x": x, "y": y, "yaw": yaw}
        except Exception:
            return None

    def _distance_to_waypoint(self, wp: Dict[str, float]) -> Optional[float]:
        pose = self._get_current_pose()
        if pose is None:
            return None
        dx = wp["x"] - pose["x"]
        dy = wp["y"] - pose["y"]
        return math.hypot(dx, dy)

    def _tick(self):
        if self.finished:
            return

        now = time.time()

        if not self.nav_ready:
            if (now - self.start_time) < self.startup_delay_sec:
                return

            self.get_logger().info("Waiting for Nav2 to become active...")
            self.navigator.waitUntilNav2Active()
            self.nav_ready = True
            self.get_logger().info("Nav2 is active.")
            return

        if self.patrol_override_active:
            return

        if self.was_paused_by_override:
            self.get_logger().info(
                f"Resuming patrol at current waypoint #{self.current_index} after IK override"
            )
            self.was_paused_by_override = False
            self.pause_until = None
            self.goal_sent = False
            self.goal_start_time = None

        if self.pause_until is not None:
            if now < self.pause_until:
                return
            self.pause_until = None
            self.goal_sent = False
            self.goal_start_time = None

        if not self.goal_sent:
            sent = self._send_current_goal()
            if sent:
                self.goal_sent = True
                self.nav_goal_active = True
                self.goal_start_time = now
            else:
                self._advance_waypoint()
            return

        if self.goal_start_time is not None:
            elapsed = now - self.goal_start_time
            if elapsed > self.goal_timeout_sec:
                self.get_logger().warn(
                    f"Goal #{self.current_index} timed out after {elapsed:.1f}s. Cancelling."
                )
                try:
                    self.navigator.cancelTask()
                except Exception:
                    pass
                self.nav_goal_active = False
                self._advance_waypoint()
                return

        if not self.navigator.isTaskComplete():
            return

        result = self.navigator.getResult()
        self._handle_result(result)

    def _send_current_goal(self) -> bool:
        wp = self.waypoints[self.current_index]
        dist = self._distance_to_waypoint(wp)

        if dist is not None:
            self.get_logger().info(
                f"Goal #{self.current_index} distance from robot: {dist:.3f} m"
            )
            if dist < self.min_waypoint_separation_m:
                self.get_logger().warn(
                    f"Skipping waypoint #{self.current_index}: too close "
                    f"(< {self.min_waypoint_separation_m:.2f} m)"
                )
                return False

        pose = self._make_pose(wp["x"], wp["y"], wp["yaw"])

        self.get_logger().info(
            f"SENDING GOAL #{self.current_index}: "
            f"x={wp['x']:.3f}, y={wp['y']:.3f}, yaw={wp['yaw']:.3f}, "
            f"ignore_goal_yaw={self.ignore_goal_yaw}"
        )
        self.navigator.goToPose(pose)
        return True

    def _handle_result(self, result):
        self.nav_goal_active = False

        self.get_logger().info(
            f"RESULT at waypoint #{self.current_index}: {result}"
        )

        if result == TaskResult.SUCCEEDED:
            self.get_logger().info(f"Reached waypoint #{self.current_index}")
        elif result == TaskResult.CANCELED:
            self.get_logger().warn(f"Waypoint #{self.current_index} was canceled")
            if self.patrol_override_active:
                return
        elif result == TaskResult.FAILED:
            self.get_logger().warn(f"Failed to reach waypoint #{self.current_index}")
        else:
            self.get_logger().warn(
                f"Unexpected navigation result at waypoint #{self.current_index}: {result}"
            )

        self._advance_waypoint()

    def _advance_waypoint(self):
        self.goal_sent = False
        self.goal_start_time = None

        if len(self.waypoints) == 1:
            self.get_logger().info("Only one waypoint configured; repeating it.")
            if self.pause_at_waypoint_sec > 0.0:
                self.pause_until = time.time() + self.pause_at_waypoint_sec
            return

        if self.ping_pong:
            next_index = self.current_index + self.direction
            if next_index >= len(self.waypoints):
                self.direction = -1
                next_index = len(self.waypoints) - 2
            elif next_index < 0:
                self.direction = 1
                next_index = 1
            self.current_index = next_index
        else:
            next_index = self.current_index + 1
            if next_index >= len(self.waypoints):
                if self.loop_forever:
                    next_index = 0
                else:
                    self.get_logger().info("Patrol complete. Stopping.")
                    self.finished = True
                    try:
                        self.navigator.cancelTask()
                    except Exception:
                        pass
                    return
            self.current_index = next_index

        if self.pause_at_waypoint_sec > 0.0:
            self.pause_until = time.time() + self.pause_at_waypoint_sec
            self.get_logger().info(
                f"Pausing {self.pause_at_waypoint_sec:.1f}s before waypoint #{self.current_index}"
            )

    def destroy_node(self):
        try:
            self.navigator.cancelTask()
            self._publish_stop()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PatrolNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Patrol node interrupted by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()