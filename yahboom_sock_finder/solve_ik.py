def solve_ik(self, x_m, y_m, z_m):
    req = ArmKinemarics.Request()

    req.tar_x = float(x_m * 1000.0)
    req.tar_y = float(y_m * 1000.0)
    req.tar_z = float(z_m * 1000.0)

 
    req.roll = float(self.ik_roll)
    req.pitch = float(self.ik_pitch)
    req.yaw = float(self.ik_yaw)

    req.cur_joint1 = float(self.current_joints[0])
    req.cur_joint2 = float(self.current_joints[1])
    req.cur_joint3 = float(self.current_joints[2])
    req.cur_joint4 = float(self.current_joints[3])
    req.cur_joint5 = float(self.current_joints[4])
    req.cur_joint6 = float(self.current_joints[5])

    req.kin_name = "ik"

    future = self.ik_client.call_async(req)
    rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)

    if not future.done() or future.result() is None:
        return False, None

    res = future.result()

    joints = [
        float(res.joint1),
        float(res.joint2),
        float(res.joint3),
        float(res.joint4),
        float(res.joint5),
        float(res.joint6),
    ]

    return True, joints