#!/usr/bin/env python3
import numpy as np
from scipy.linalg import solve_continuous_are

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


# ============================================================
# FR3 3DOF exact LQR around standing pose
# Active joints: 3,4,5
# Held joints:   1,2,6,7
# ============================================================

# standing pose for held joints
Q1_FIXED = 0.0
Q2_FIXED = -0.7853981634
Q6_FIXED = 1.5707963268
Q7_FIXED = 0.7853981634


def get_full_fr3_state(q345, dq345):
    q_full = np.array([
        Q1_FIXED,
        Q2_FIXED,
        q345[0],
        q345[1],
        q345[2],
        Q6_FIXED,
        Q7_FIXED
    ], dtype=float)

    dq_full = np.array([
        0.0,
        0.0,
        dq345[0],
        dq345[1],
        dq345[2],
        0.0,
        0.0
    ], dtype=float)

    return q_full, dq_full


# -------------------------
# Placeholder full FR3 dynamics approximation
# Replace with exact FR3 M,C,G later if desired.
# For now: decoupled local model around standing pose.
# -------------------------
B_DAMP = np.diag([0.2, 0.2, 0.2]).astype(float)
M_LOCAL = np.diag([1.5, 1.8, 1.2]).astype(float)


def reduced_f(x, u):
    q = x[0:3]
    dq = x[3:6]
    tau = np.asarray(u, dtype=float).reshape(3)

    qdd = np.linalg.solve(M_LOCAL, tau - B_DAMP @ dq)

    return np.array([
        dq[0], dq[1], dq[2],
        qdd[0], qdd[1], qdd[2]
    ], dtype=float)


def equilibrium_torque(q345):
    # with held standing-pose controller in background, start from zero local torque
    return np.zeros(3, dtype=float)


def linearize_numerical(f, x0, u0, epsx=1e-6, epsu=1e-6):
    n = x0.size
    m = u0.size
    A = np.zeros((n, n), dtype=float)
    B = np.zeros((n, m), dtype=float)

    for i in range(n):
        dx = np.zeros(n, dtype=float)
        dx[i] = epsx
        A[:, i] = (f(x0 + dx, u0) - f(x0 - dx, u0)) / (2.0 * epsx)

    for j in range(m):
        du = np.zeros(m, dtype=float)
        du[j] = epsu
        B[:, j] = (f(x0, u0 + du) - f(x0, u0 - du)) / (2.0 * epsu)

    return A, B


class FrankaJ345ExactLQR(Node):
    def __init__(self):
        super().__init__('fr3_j345_exact_lqr')

        self.q_des = np.array([0.0, -2.2, 0.0], dtype=float)
        self.dq_des = np.zeros(3, dtype=float)
        self.x_eq = np.hstack((self.q_des, self.dq_des))
        self.u_eq = equilibrium_torque(self.q_des)

        self.A, self.B = linearize_numerical(reduced_f, self.x_eq, self.u_eq)

        self.Q = np.diag([120.0, 120.0, 80.0, 10.0, 10.0, 10.0])
        self.R = np.diag([0.8, 0.8, 0.8])

        P = solve_continuous_are(self.A, self.B, self.Q, self.R)
        self.K = np.linalg.solve(self.R, self.B.T @ P)

        self.rate_hz = 200.0
        self.x = None
        self.print_count = 0

        self.get_logger().info(f"q_des = {self.q_des}")
        self.get_logger().info(f"u_eq  = {self.u_eq}")
        self.get_logger().info(f"K =\n{self.K}")

        self.sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.cb_joint_state,
            10
        )

        self.pub = self.create_publisher(
            Float64MultiArray,
            '/arm_3dof_cmd/commands',
            10
        )

        self.timer = self.create_timer(1.0 / self.rate_hz, self.loop)

    def cb_joint_state(self, msg: JointState):
        try:
            i3 = msg.name.index('fr3_joint3')
            i4 = msg.name.index('fr3_joint4')
            i5 = msg.name.index('fr3_joint5')
        except ValueError:
            return

        self.x = np.array([
            msg.position[i3],
            msg.position[i4],
            msg.position[i5],
            msg.velocity[i3],
            msg.velocity[i4],
            msg.velocity[i5]
        ], dtype=float)

    def loop(self):
        if self.x is None:
            return

        tau = self.u_eq - self.K @ (self.x - self.x_eq)

        # clip for safety in sim
        tau = np.clip(tau, -10.0, 10.0)

        self.print_count += 1
        if self.print_count % 100 == 0:
            self.get_logger().info(
                f"q=[{self.x[0]:.4f}, {self.x[1]:.4f}, {self.x[2]:.4f}] "
                f"dq=[{self.x[3]:.4f}, {self.x[4]:.4f}, {self.x[5]:.4f}] "
                f"tau=[{tau[0]:.4f}, {tau[1]:.4f}, {tau[2]:.4f}]"
            )

        msg = Float64MultiArray()
        msg.data = [float(tau[0]), float(tau[1]), float(tau[2])]
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = FrankaJ345ExactLQR()
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
