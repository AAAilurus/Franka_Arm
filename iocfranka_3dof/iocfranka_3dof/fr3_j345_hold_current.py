#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


class HoldCurrent(Node):
    def __init__(self):
        super().__init__('fr3_j345_hold_current')

        self.declare_parameter('joint_name_prefix', 'follower_fr3_')
        self.declare_parameter('joint_states_topic', '/follower/joint_states')
        self.declare_parameter('command_topic', '/follower/arm_3dof_cmd/commands')
        self.declare_parameter('rate_hz', 500.0)

        self.declare_parameter('kp', 6.0)
        self.declare_parameter('kd', 8.0)
        self.declare_parameter('tau_limit', 2.0)

        prefix = self.get_parameter('joint_name_prefix').value
        self.topic_state = self.get_parameter('joint_states_topic').value
        self.topic_cmd = self.get_parameter('command_topic').value
        self.rate_hz = float(self.get_parameter('rate_hz').value)

        self.kp = float(self.get_parameter('kp').value)
        self.kd = float(self.get_parameter('kd').value)
        self.tau_limit = float(self.get_parameter('tau_limit').value)

        self.joints = [f'{prefix}joint3', f'{prefix}joint4', f'{prefix}joint5']

        self.x = None
        self.q_hold = None
        self.count = 0

        self.sub = self.create_subscription(JointState, self.topic_state, self.cb, 10)
        self.pub = self.create_publisher(Float64MultiArray, self.topic_cmd, 10)
        self.timer = self.create_timer(1.0 / self.rate_hz, self.loop)

        self.get_logger().info('========== HOLD CURRENT J345 ==========')
        self.get_logger().info(f'joints={self.joints}')
        self.get_logger().info(f'sub={self.topic_state}')
        self.get_logger().info(f'pub={self.topic_cmd}')
        self.get_logger().info(f'kp={self.kp}, kd={self.kd}, tau_limit={self.tau_limit}')

    def cb(self, msg):
        try:
            idx = [msg.name.index(j) for j in self.joints]
        except ValueError:
            return

        self.x = np.array([
            msg.position[idx[0]],
            msg.position[idx[1]],
            msg.position[idx[2]],
            msg.velocity[idx[0]],
            msg.velocity[idx[1]],
            msg.velocity[idx[2]],
        ], dtype=float)

        if self.q_hold is None:
            self.q_hold = self.x[:3].copy()
            self.get_logger().info(f'LOCKED q_hold={self.q_hold}')

    def loop(self):
        if self.x is None or self.q_hold is None:
            return

        q = self.x[:3]
        dq = self.x[3:6]

        tau = self.kp * (self.q_hold - q) - self.kd * dq
        tau = np.clip(tau, -self.tau_limit, self.tau_limit)

        msg = Float64MultiArray()
        msg.data = tau.tolist()
        self.pub.publish(msg)

        self.count += 1
        if self.count % int(self.rate_hz * 0.5) == 0:
            self.get_logger().info(
                f'q=[{q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}] '
                f'dq=[{dq[0]:.4f}, {dq[1]:.4f}, {dq[2]:.4f}] '
                f'tau=[{tau[0]:.4f}, {tau[1]:.4f}, {tau[2]:.4f}]'
            )


def main(args=None):
    rclpy.init(args=args)
    node = HoldCurrent()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            zero = Float64MultiArray()
            zero.data = [0.0, 0.0, 0.0]
            for _ in range(20):
                node.pub.publish(zero)
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
