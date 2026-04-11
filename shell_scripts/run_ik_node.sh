ros2 topic pub /arm6_joints arm_msgs/msg/ArmJoints "{joint1: 90, joint2: 90, joint3: 45, joint4: 0, joint5: 90, joint6: 90, time: 1500}" --once

ros2 run yahboom_sock_finder move_to_sock_mask_ik_node
