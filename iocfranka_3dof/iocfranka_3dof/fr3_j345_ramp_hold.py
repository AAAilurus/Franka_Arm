#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


class RampHold(Node):
    def __init__(self):
        super().__init__('fr3_j345_ramp_hold')

        self.declare_parameter('joint_name_prefix', 'follower_fr3_')
        self.declare_parameter('joint_states_topic', '/follower/joint_states')
        self.declare_parameter('command_topic', '/follower/arm_3dof_cmd/commands')
        self.declare_parameter('rate_hz', 500.0)

        self.declare_parameter('q_des_j3', 0.3)
        self.declare_parameter('q_des_j4', -1.8)
        self.declare_parameter('q_des_j5', 0.2)

        self.declare_parameter('kp', 4.0)
        self.declare_parameter('kd', 10.0)
        self.declare_parameter('tau_limit', 1.0)
        self.declare_parameter('max_ref_speed', 0.15)  # rad/s, slow reference motion

        prefix = self.get_parameter('joint_name_prefix').value
        self.topic_state = self.get_parameter('joint_states_topic').value
        self.topic_cmd = self.get_parameter('command_topic').value
        self.rate_hz = float(self.get_parameter('rate_hz').value)
        self.dt = 1.0 / self.rate_hz

        self.kp = float(self.get_parameter('kp').value)
        self.kd = float(self.get_parameter('kd').value)
        self.tau_limit = float(self.get_parameter('tau_limit').value)
        self.max_ref_speed = float(self.get_parameter('max_ref_speed').value)

        self.joints = [f'{prefix}joint3', f'{prefix}joint4', f'{prefix}joint5']

        self.q_des = np.array([
            float(self.get_parameter('q_des_j3').value),
            float(self.get_parameter('q_des_j4').value),
            float(self.get_parameter('q_des_j5').value),
        ], dtype=float)

        self.x = None
        self.q_ref = None
        self.count = 0

        self.sub = self.create_subscription(JointState, self.topic_state, self.cb, 10)
        self.pub = self.create_publisher(Float64MultiArray, self.topic_cmd, 10)
        self.timer = self.create_timer(self.dt, self.loop)

        self.get_logger().info('========== RAMP HOLD J345 ==========')
        self.get_logger().info(f'joints={self.joints}')
        self.get_logger().info(f'q_des={self.q_des}')
        self.get_logger().info(f'kp={self.kp}, kd={self.kd}, tau_limit={self.tau_limit}, max_ref_speed={self.max_ref_speed}')

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

        if self.q_ref is None:
            self.q_ref = self.x[:3].copy()
            self.get_logger().info(f'INITIAL q_ref={self.q_ref}')

    def loop(self):
        if self.x is None or self.q_ref is None:
            return

        q = self.x[:3]
        dq = self.x[3:6]

        # Slowly move internal reference toward target.
        diff = self.q_des - self.q_ref
        max_step = self.max_ref_speed * self.dt
        step = np.clip(diff, -max_step, max_step)
        self.q_ref = self.q_ref + step

        # PD around the slowly moving reference.
        tau = self.kp * (self.q_ref - q) - self.kd * dq
        tau = np.clip(tau, -self.tau_limit, self.tau_limit)

        msg = Float64MultiArray()
        msg.data = tau.tolist()
        self.pub.publish(msg)

        self.count += 1
        if self.count % int(self.rate_hz * 0.5) == 0:
            err = self.q_des - q
            self.get_logger().info(
                f'q=[{q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}] '
                f'q_ref=[{self.q_ref[0]:.4f}, {self.q_ref[1]:.4f}, {self.q_ref[2]:.4f}] '
                f'dq=[{dq[0]:.4f}, {dq[1]:.4f}, {dq[2]:.4f}] '
                f'err=[{err[0]:.4f}, {err[1]:.4f}, {err[2]:.4f}] '
                f'tau=[{tau[0]:.4f}, {tau[1]:.4f}, {tau[2]:.4f}]'
            )


def main(args=None):
    rclpy.init(args=args)
    node = RampHold()
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
