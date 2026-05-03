#!/usr/bin/env python3
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


class FR3LeaderTorqueRun(Node):
    def __init__(self):
        super().__init__('fr3_leader_torque_run')

        self.declare_parameter('joint_name_prefix', 'leader_fr3_')
        self.declare_parameter('joint_states_topic', '/leader/joint_states')
        self.declare_parameter('command_topic', '/leader/arm_3dof_cmd/commands')
        self.declare_parameter('rate_hz', 200.0)

        self.declare_parameter('q_des_j3', 0.3)
        self.declare_parameter('q_des_j4', -1.8)
        self.declare_parameter('q_des_j5', 0.2)

        self.declare_parameter('kfile', '/tmp/fr3_torque_K.npz')
        self.declare_parameter('scale', 1.0)
        self.declare_parameter('tau_limit', 5.0)

        prefix = str(self.get_parameter('joint_name_prefix').value)
        self.js_topic = str(self.get_parameter('joint_states_topic').value)
        self.cmd_topic = str(self.get_parameter('command_topic').value)
        self.rate_hz = float(self.get_parameter('rate_hz').value)

        self.q_des = np.array([
            float(self.get_parameter('q_des_j3').value),
            float(self.get_parameter('q_des_j4').value),
            float(self.get_parameter('q_des_j5').value),
        ], dtype=float)

        self.kfile = str(self.get_parameter('kfile').value)
        self.scale = float(self.get_parameter('scale').value)
        self.tau_limit = float(self.get_parameter('tau_limit').value)

        self.K = np.load(self.kfile)['K']  # 3x6

        if self.K.shape != (3, 6):
            raise RuntimeError(f"K must be 3x6, got {self.K.shape}")

        self.joints = [
            f'{prefix}joint3' if prefix.endswith('fr3_') else f'{prefix}fr3_joint3',
            f'{prefix}joint4' if prefix.endswith('fr3_') else f'{prefix}fr3_joint4',
            f'{prefix}joint5' if prefix.endswith('fr3_') else f'{prefix}fr3_joint5',
        ]

        self.x = None

        self.sub = self.create_subscription(JointState, self.js_topic, self.cb_js, 10)
        self.pub = self.create_publisher(Float64MultiArray, self.cmd_topic, 10)
        self.timer = self.create_timer(1.0 / self.rate_hz, self.tick)

        self.get_logger().info("========== FR3 learned torque runner ==========")
        self.get_logger().info(f"K file = {self.kfile}")
        self.get_logger().info(f"K =\n{self.K}")
        self.get_logger().info(f"joints={self.joints}")
        self.get_logger().info(f"sub={self.js_topic}")
        self.get_logger().info(f"pub={self.cmd_topic}")
        self.get_logger().info(f"q_des={self.q_des}")
        self.get_logger().info(f"scale={self.scale}, tau_limit={self.tau_limit}")

    def cb_js(self, msg):
        try:
            idx = [msg.name.index(j) for j in self.joints]
        except ValueError:
            return

        q = np.array([
            msg.position[idx[0]],
            msg.position[idx[1]],
            msg.position[idx[2]],
        ], dtype=float)

        dq = np.array([
            msg.velocity[idx[0]],
            msg.velocity[idx[1]],
            msg.velocity[idx[2]],
        ], dtype=float)

        e = q - self.q_des
        self.x = np.hstack((e, dq))

    def tick(self):
        if self.x is None:
            return

        tau = -self.scale * (self.K @ self.x)
        tau = np.clip(tau, -self.tau_limit, self.tau_limit)

        msg = Float64MultiArray()
        msg.data = [float(tau[0]), float(tau[1]), float(tau[2])]
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = FR3LeaderTorqueRun()
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


if __name__ == '__main__':
    main()
