#!/usr/bin/env python3
import os
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


def make_joint_names(prefix):
    if prefix.endswith("fr3_"):
        return [f"{prefix}joint3", f"{prefix}joint4", f"{prefix}joint5"]
    return [f"{prefix}fr3_joint3", f"{prefix}fr3_joint4", f"{prefix}fr3_joint5"]


class FR3FMFollowerRun(Node):
    def __init__(self):
        super().__init__("fr3_fm_follower_run")

        self.declare_parameter("joint_name_prefix", "follower_fr3_")
        self.declare_parameter("joint_states_topic", "/follower/joint_states")
        self.declare_parameter("command_topic", "/follower/arm_3dof_cmd/commands")

        self.declare_parameter("q_des_j3", 0.3)
        self.declare_parameter("q_des_j4", -1.8)
        self.declare_parameter("q_des_j5", 0.2)

        self.declare_parameter("rate_hz", 200.0)
        self.declare_parameter("tau_limit", 5.0)
        self.declare_parameter("scale", 1.0)
        self.declare_parameter("k_path", "/tmp/fr3_freemodel_learned/K_learned.npy")

        prefix = str(self.get_parameter("joint_name_prefix").value)
        self.joints = make_joint_names(prefix)

        self.joint_states_topic = str(self.get_parameter("joint_states_topic").value)
        self.command_topic = str(self.get_parameter("command_topic").value)

        self.q_des = np.array([
            float(self.get_parameter("q_des_j3").value),
            float(self.get_parameter("q_des_j4").value),
            float(self.get_parameter("q_des_j5").value),
        ], dtype=float)

        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.dt = 1.0 / self.rate_hz
        self.tau_limit = float(self.get_parameter("tau_limit").value)
        self.scale = float(self.get_parameter("scale").value)
        self.k_path = str(self.get_parameter("k_path").value)

        if not os.path.exists(self.k_path):
            raise RuntimeError(f"K file not found: {self.k_path}")

        self.K = np.load(self.k_path)
        if self.K.shape != (3, 6):
            raise RuntimeError(f"K must be 3x6, got {self.K.shape}")

        self.x = None

        self.sub = self.create_subscription(JointState, self.joint_states_topic, self.cb_js, 10)
        self.pub = self.create_publisher(Float64MultiArray, self.command_topic, 10)
        self.timer = self.create_timer(self.dt, self.step)

        self.get_logger().info("========== FR3 freemodel follower learned run ==========")
        self.get_logger().info(f"joints={self.joints}")
        self.get_logger().info(f"sub={self.joint_states_topic}")
        self.get_logger().info(f"pub={self.command_topic}")
        self.get_logger().info(f"q_des={self.q_des}")
        self.get_logger().info(f"k_path={self.k_path}")
        self.get_logger().info(f"K=\n{np.round(self.K, 4)}")

    def cb_js(self, msg):
        try:
            idx = [msg.name.index(j) for j in self.joints]
        except ValueError:
            return

        q = np.array([msg.position[i] for i in idx], dtype=float)
        dq = np.array([msg.velocity[i] if len(msg.velocity) > i else 0.0 for i in idx], dtype=float)

        e = q - self.q_des
        self.x = np.hstack((e, dq))

    def step(self):
        if self.x is None:
            return

        # Learned policy:
        # u = -K_learned e
        tau = -self.scale * (self.K @ self.x)
        tau = np.clip(tau, -self.tau_limit, self.tau_limit)

        msg = Float64MultiArray()
        msg.data = [float(tau[0]), float(tau[1]), float(tau[2])]
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = FR3FMFollowerRun()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
