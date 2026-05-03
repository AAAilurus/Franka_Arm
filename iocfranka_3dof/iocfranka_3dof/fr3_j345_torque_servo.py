#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


class FR3J345TorqueServo(Node):
    def __init__(self):
        super().__init__('fr3_j345_torque_servo')

        self.declare_parameter('joint_name_prefix', '')
        self.declare_parameter('joint_states_topic', '/joint_states')
        self.declare_parameter('command_topic', '/arm_3dof_cmd/commands')
        self.declare_parameter('rate_hz', 200.0)

        self.declare_parameter('q_des_j3', 0.3)
        self.declare_parameter('q_des_j4', -1.8)
        self.declare_parameter('q_des_j5', 0.2)

        self.declare_parameter('kp_j3', 1.8)
        self.declare_parameter('kp_j4', 2.2)
        self.declare_parameter('kp_j5', 1.2)
        self.declare_parameter('kd_j3', 1.2)
        self.declare_parameter('kd_j4', 1.4)
        self.declare_parameter('kd_j5', 0.8)

        self.declare_parameter('tau_limit', 1.2)
        self.declare_parameter('deadband_pos', 0.015)
        self.declare_parameter('deadband_vel', 0.03)

        prefix = str(self.get_parameter('joint_name_prefix').value)
        joint_states_topic = str(self.get_parameter('joint_states_topic').value)
        command_topic = str(self.get_parameter('command_topic').value)
        self.rate_hz = float(self.get_parameter('rate_hz').value)

        self.joints = [
            f'{prefix}joint3' if prefix.endswith('fr3_') else f'{prefix}fr3_joint3',
            f'{prefix}joint4' if prefix.endswith('fr3_') else f'{prefix}fr3_joint4',
            f'{prefix}joint5' if prefix.endswith('fr3_') else f'{prefix}fr3_joint5',
        ]

        self.q_des = np.array([
            float(self.get_parameter('q_des_j3').value),
            float(self.get_parameter('q_des_j4').value),
            float(self.get_parameter('q_des_j5').value),
        ])

        self.kp = np.array([
            float(self.get_parameter('kp_j3').value),
            float(self.get_parameter('kp_j4').value),
            float(self.get_parameter('kp_j5').value),
        ])
        self.kd = np.array([
            float(self.get_parameter('kd_j3').value),
            float(self.get_parameter('kd_j4').value),
            float(self.get_parameter('kd_j5').value),
        ])

        self.tau_limit = float(self.get_parameter('tau_limit').value)
        self.deadband_pos = float(self.get_parameter('deadband_pos').value)
        self.deadband_vel = float(self.get_parameter('deadband_vel').value)

        self.x = None
        self.print_count = 0

        self.sub = self.create_subscription(JointState, joint_states_topic, self.cb, 10)
        self.pub = self.create_publisher(Float64MultiArray, command_topic, 10)
        self.timer = self.create_timer(1.0 / self.rate_hz, self.loop)

        self.get_logger().info('========== FR3 J345 SIMPLE TORQUE SERVO ==========')
        self.get_logger().info(f'joints = {self.joints}')
        self.get_logger().info(f'q_des = {self.q_des}')
        self.get_logger().info(f'kp = {self.kp}, kd = {self.kd}, tau_limit = {self.tau_limit}')

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

    def loop(self):
        if self.x is None:
            return

        q = self.x[:3]
        dq = self.x[3:6]
        e = self.q_des - q

        tau = self.kp * e - self.kd * dq

        # When very close, only damp velocity. This prevents "searching".
        close = (np.abs(e) < self.deadband_pos) & (np.abs(dq) < self.deadband_vel)
        tau[close] = -self.kd[close] * dq[close]

        tau = np.clip(tau, -self.tau_limit, self.tau_limit)

        msg = Float64MultiArray()
        msg.data = tau.tolist()
        self.pub.publish(msg)

        self.print_count += 1
        if self.print_count % int(max(self.rate_hz * 0.5, 1)) == 0:
            self.get_logger().info(
                'q=[%.4f, %.4f, %.4f] dq=[%.4f, %.4f, %.4f] e=[%.4f, %.4f, %.4f] tau=[%.4f, %.4f, %.4f]' %
                (q[0], q[1], q[2], dq[0], dq[1], dq[2], e[0], e[1], e[2], tau[0], tau[1], tau[2])
            )


def main(args=None):
    rclpy.init(args=args)
    node = FR3J345TorqueServo()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        zero = Float64MultiArray()
        zero.data = [0.0, 0.0, 0.0]
        for _ in range(20):
            node.pub.publish(zero)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
