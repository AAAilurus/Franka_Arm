#!/usr/bin/env python3
import math
import numpy as np
from scipy.linalg import solve_continuous_are

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


# ============================================================
# FR3 3DOF MATLAB-style LQR
# Active joints: 3,4,5
# Held joints:   1,2,6,7
# ============================================================

Q1_FIXED = 0.0
Q2_FIXED = -0.7853981634
Q6_FIXED = 1.5707963268
Q7_FIXED = 0.7853981634


class Params:
    pass

p = Params()

# ----- active link 1 (joint3 segment) -----
p.l1  = 0.3160
p.m1  = 2.2449013699
p.lc1 = 0.05019253061090138
p.I1  = 0.019044494482244823

# ----- active link 2 (joint4 segment) -----
p.l2  = 0.3840
p.m2  = 2.6155955791
p.lc2 = 0.0784571246941843
p.I2  = 0.04125471171146641

# ----- active link 3 (joint5 segment) -----
# keep separate so you can refine from FR3 inertials later
p.l3  = 0.1070
p.m3  = 2.3271207594
p.lc3 = 0.10161188666736627
p.I3  = 0.016423625579357254

# damping
p.b1 = 0.003
p.b2 = 0.003
p.b3 = 0.003

# gravity
p.g = 9.81


def mass_matrix_3r(q, p):
    q1, q2, q3 = float(q[0]), float(q[1]), float(q[2])

    Jv1 = np.array([
        [-p.lc1 * math.sin(q1), 0.0, 0.0],
        [ p.lc1 * math.cos(q1), 0.0, 0.0]
    ], dtype=float)

    Jv2 = np.array([
        [-p.l1 * math.sin(q1) - p.lc2 * math.sin(q1 + q2),
         -p.lc2 * math.sin(q1 + q2),
          0.0],
        [ p.l1 * math.cos(q1) + p.lc2 * math.cos(q1 + q2),
          p.lc2 * math.cos(q1 + q2),
          0.0]
    ], dtype=float)

    Jv3 = np.array([
        [-p.l1 * math.sin(q1) - p.l2 * math.sin(q1 + q2) - p.lc3 * math.sin(q1 + q2 + q3),
         -p.l2 * math.sin(q1 + q2) - p.lc3 * math.sin(q1 + q2 + q3),
         -p.lc3 * math.sin(q1 + q2 + q3)],
        [ p.l1 * math.cos(q1) + p.l2 * math.cos(q1 + q2) + p.lc3 * math.cos(q1 + q2 + q3),
          p.l2 * math.cos(q1 + q2) + p.lc3 * math.cos(q1 + q2 + q3),
          p.lc3 * math.cos(q1 + q2 + q3)]
    ], dtype=float)

    Jw1 = np.array([[1.0, 0.0, 0.0]], dtype=float)
    Jw2 = np.array([[1.0, 1.0, 0.0]], dtype=float)
    Jw3 = np.array([[1.0, 1.0, 1.0]], dtype=float)

    M = (
        p.m1 * (Jv1.T @ Jv1) + p.I1 * (Jw1.T @ Jw1) +
        p.m2 * (Jv2.T @ Jv2) + p.I2 * (Jw2.T @ Jw2) +
        p.m3 * (Jv3.T @ Jv3) + p.I3 * (Jw3.T @ Jw3)
    )
    return M


def potential_energy(q, p):
    q1, q2, q3 = float(q[0]), float(q[1]), float(q[2])

    y1 = p.lc1 * math.sin(q1)
    y2 = p.l1 * math.sin(q1) + p.lc2 * math.sin(q1 + q2)
    y3 = p.l1 * math.sin(q1) + p.l2 * math.sin(q1 + q2) + p.lc3 * math.sin(q1 + q2 + q3)

    return p.g * (p.m1 * y1 + p.m2 * y2 + p.m3 * y3)


def gravity_vector_3r(q, p):
    epsg = 1e-7
    G = np.zeros(3, dtype=float)
    for i in range(3):
        dq = np.zeros(3, dtype=float)
        dq[i] = epsg
        G[i] = (potential_energy(q + dq, p) - potential_energy(q - dq, p)) / (2.0 * epsg)
    return G


def coriolis_matrix_numeric(q, dq, p):
    n = 3
    C = np.zeros((n, n), dtype=float)
    epsc = 1e-7
    dM = np.zeros((n, n, n), dtype=float)

    for k in range(n):
        dqk = np.zeros(n, dtype=float)
        dqk[k] = epsc
        Mplus = mass_matrix_3r(q + dqk, p)
        Mminus = mass_matrix_3r(q - dqk, p)
        dM[:, :, k] = (Mplus - Mminus) / (2.0 * epsc)

    for i in range(n):
        for j in range(n):
            cij = 0.0
            for k in range(n):
                cijk = 0.5 * (dM[i, j, k] + dM[i, k, j] - dM[j, k, i])
                cij += cijk * dq[k]
            C[i, j] = cij
    return C


def arm_matrices(q, dq, p):
    M = mass_matrix_3r(q, p)
    C = coriolis_matrix_numeric(q, dq, p)
    G = gravity_vector_3r(q, p)
    Bv = np.diag([p.b1, p.b2, p.b3]).astype(float)
    return M, C, G, Bv


def arm_f(x, u, p):
    q = x[0:3]
    dq = x[3:6]
    tau = np.asarray(u, dtype=float).reshape(3)

    M, C, G, Bv = arm_matrices(q, dq, p)
    qdd = np.linalg.solve(M, tau - C @ dq - G - Bv @ dq)

    return np.array([dq[0], dq[1], dq[2], qdd[0], qdd[1], qdd[2]], dtype=float)


def equilibrium_torque(q, p):
    dq = np.zeros(3, dtype=float)
    _, _, G, _ = arm_matrices(q, dq, p)
    return G.copy()


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
        super().__init__('fr3_j345_matlab_lqr_follower')

        self.declare_parameter('joint_name_prefix', '')
        self.declare_parameter('joint_states_topic', '/joint_states')
        self.declare_parameter('command_topic', '/arm_3dof_cmd/commands')
        self.declare_parameter('rate_hz', 200.0)
        self.declare_parameter('tau_limit', 25.0)
        self.declare_parameter('q_des_j3', 0.0)
        self.declare_parameter('q_des_j4', -2.2)
        self.declare_parameter('q_des_j5', 0.0)

        prefix = self.get_parameter('joint_name_prefix').value
        joint_states_topic = self.get_parameter('joint_states_topic').value
        command_topic = self.get_parameter('command_topic').value
        self.rate_hz = float(self.get_parameter('rate_hz').value)
        self.tau_limit = float(self.get_parameter('tau_limit').value)
        q_des_j3 = float(self.get_parameter('q_des_j3').value)
        q_des_j4 = float(self.get_parameter('q_des_j4').value)
        q_des_j5 = float(self.get_parameter('q_des_j5').value)

        self.j3 = f'{prefix}fr3_joint3' if prefix and not prefix.endswith('fr3_') else (f'{prefix}joint3' if prefix.endswith('fr3_') else 'fr3_joint3')
        self.j4 = f'{prefix}fr3_joint4' if prefix and not prefix.endswith('fr3_') else (f'{prefix}joint4' if prefix.endswith('fr3_') else 'fr3_joint4')
        self.j5 = f'{prefix}fr3_joint5' if prefix and not prefix.endswith('fr3_') else (f'{prefix}joint5' if prefix.endswith('fr3_') else 'fr3_joint5')

        self.q_des = np.array([q_des_j3, q_des_j4, q_des_j5], dtype=float)
        self.dq_des = np.zeros(3, dtype=float)
        self.x_eq = np.hstack((self.q_des, self.dq_des))

        # MATLAB-style torque-control equilibrium:
        # q_des is only a true equilibrium if u_eq balances gravity.
        self.u_eq = equilibrium_torque(self.q_des, p)

        self.A, self.B = linearize_numerical(lambda x, u: arm_f(x, u, p), self.x_eq, self.u_eq)

        self.Q = np.diag([100.0, 100.0, 100.0, 10.0, 10.0, 10.0])
        self.R = np.diag([0.8, 0.8, 0.8])

        P = solve_continuous_are(self.A, self.B, self.Q, self.R)
        self.K = np.linalg.solve(self.R, self.B.T @ P)

        self.x = None
        self.print_count = 0

        self.get_logger().info('========== FR3 3DOF MATLAB-style LQR FOLLOWER ==========')
        self.get_logger().info(f'joint_states_topic = {joint_states_topic}')
        self.get_logger().info(f'command_topic      = {command_topic}')
        self.get_logger().info(f'active joints      = {[self.j3, self.j4, self.j5]}')
        self.get_logger().info(f'q_des             = {self.q_des}')
        self.get_logger().info(f'u_eq              = {self.u_eq}')
        self.get_logger().info(f'tau_limit         = {self.tau_limit}')
        self.get_logger().info(f'A =\\n{self.A}')
        self.get_logger().info(f'B =\\n{self.B}')
        self.get_logger().info(f'K =\\n{self.K}')

        self.sub = self.create_subscription(JointState, joint_states_topic, self.cb_joint_state, 10)
        self.pub = self.create_publisher(Float64MultiArray, command_topic, 10)
        self.timer = self.create_timer(1.0 / self.rate_hz, self.loop)

    def cb_joint_state(self, msg: JointState):
        try:
            i3 = msg.name.index(self.j3)
            i4 = msg.name.index(self.j4)
            i5 = msg.name.index(self.j5)
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
        tau = np.clip(tau, -self.tau_limit, self.tau_limit)

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
