#!/usr/bin/env python3
import math
import time
import csv
import numpy as np
from scipy.linalg import solve_continuous_are

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


class Params:
    pass


p = Params()

# Same FR3 3DOF parameters as current exact LQR
p.l1  = 0.3160
p.m1  = 2.2449013699
p.lc1 = 0.05019253061090138
p.I1  = 0.019044494482244823

p.l2  = 0.3840
p.m2  = 2.6155955791
p.lc2 = 0.0784571246941843
p.I2  = 0.04125471171146641

p.l3  = 0.1070
p.m3  = 2.3271207594
p.lc3 = 0.10161188666736627
p.I3  = 0.016423625579357254

p.b1 = 0.003
p.b2 = 0.003
p.b3 = 0.003
p.g = 9.81


def now_str():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


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

    return (
        p.m1 * (Jv1.T @ Jv1) + p.I1 * (Jw1.T @ Jw1) +
        p.m2 * (Jv2.T @ Jv2) + p.I2 * (Jw2.T @ Jw2) +
        p.m3 * (Jv3.T @ Jv3) + p.I3 * (Jw3.T @ Jw3)
    )


def potential_energy(q, p):
    q1, q2, q3 = float(q[0]), float(q[1]), float(q[2])
    y1 = p.lc1 * math.sin(q1)
    y2 = p.l1 * math.sin(q1) + p.lc2 * math.sin(q1 + q2)
    y3 = p.l1 * math.sin(q1) + p.l2 * math.sin(q1 + q2) + p.lc3 * math.sin(q1 + q2 + q3)
    return p.g * (p.m1 * y1 + p.m2 * y2 + p.m3 * y3)


def gravity_vector_3r(q, p):
    eps = 1e-7
    G = np.zeros(3)
    for i in range(3):
        dq = np.zeros(3)
        dq[i] = eps
        G[i] = (potential_energy(q + dq, p) - potential_energy(q - dq, p)) / (2.0 * eps)
    return G


def coriolis_matrix_numeric(q, dq, p):
    n = 3
    C = np.zeros((n, n))
    eps = 1e-7
    dM = np.zeros((n, n, n))

    for k in range(n):
        e = np.zeros(n)
        e[k] = eps
        dM[:, :, k] = (mass_matrix_3r(q + e, p) - mass_matrix_3r(q - e, p)) / (2.0 * eps)

    for i in range(n):
        for j in range(n):
            val = 0.0
            for k in range(n):
                cijk = 0.5 * (dM[i, j, k] + dM[i, k, j] - dM[j, k, i])
                val += cijk * dq[k]
            C[i, j] = val
    return C


def arm_matrices(q, dq, p):
    M = mass_matrix_3r(q, p)
    C = coriolis_matrix_numeric(q, dq, p)
    G = gravity_vector_3r(q, p)
    Bv = np.diag([p.b1, p.b2, p.b3])
    return M, C, G, Bv


def arm_f(x, u, p):
    q = x[0:3]
    dq = x[3:6]
    tau = np.asarray(u, dtype=float).reshape(3)

    M, C, G, Bv = arm_matrices(q, dq, p)
    qdd = np.linalg.solve(M, tau - C @ dq - G - Bv @ dq)
    return np.array([dq[0], dq[1], dq[2], qdd[0], qdd[1], qdd[2]], dtype=float)


def linearize_numerical(f, x0, u0, epsx=1e-6, epsu=1e-6):
    n = x0.size
    m = u0.size
    A = np.zeros((n, n))
    B = np.zeros((n, m))

    for i in range(n):
        dx = np.zeros(n)
        dx[i] = epsx
        A[:, i] = (f(x0 + dx, u0) - f(x0 - dx, u0)) / (2.0 * epsx)

    for j in range(m):
        du = np.zeros(m)
        du[j] = epsu
        B[:, j] = (f(x0, u0 + du) - f(x0, u0 - du)) / (2.0 * epsu)

    return A, B


class FR3LeaderTorqueLog(Node):
    def __init__(self):
        super().__init__('fr3_leader_torque_log')

        self.declare_parameter('joint_name_prefix', 'leader_fr3_')
        self.declare_parameter('joint_states_topic', '/leader/joint_states')
        self.declare_parameter('command_topic', '/leader/arm_3dof_cmd/commands')
        self.declare_parameter('rate_hz', 200.0)
        self.declare_parameter('duration_s', 12.0)

        self.declare_parameter('q_des_j3', 0.3)
        self.declare_parameter('q_des_j4', -1.8)
        self.declare_parameter('q_des_j5', 0.2)
        self.declare_parameter('tau_limit', 5.0)

        prefix = str(self.get_parameter('joint_name_prefix').value)
        self.js_topic = str(self.get_parameter('joint_states_topic').value)
        self.cmd_topic = str(self.get_parameter('command_topic').value)
        self.rate_hz = float(self.get_parameter('rate_hz').value)
        self.duration_s = float(self.get_parameter('duration_s').value)
        self.tau_limit = float(self.get_parameter('tau_limit').value)

        q_des = [
            float(self.get_parameter('q_des_j3').value),
            float(self.get_parameter('q_des_j4').value),
            float(self.get_parameter('q_des_j5').value),
        ]

        self.joints = [
            f'{prefix}joint3' if prefix.endswith('fr3_') else f'{prefix}fr3_joint3',
            f'{prefix}joint4' if prefix.endswith('fr3_') else f'{prefix}fr3_joint4',
            f'{prefix}joint5' if prefix.endswith('fr3_') else f'{prefix}fr3_joint5',
        ]

        self.q_des = np.array(q_des, dtype=float)
        self.x_eq = np.hstack((self.q_des, np.zeros(3)))
        self.u_eq = np.zeros(3)

        A, B = linearize_numerical(lambda x, u: arm_f(x, u, p), self.x_eq, self.u_eq)
        Q = np.diag([120.0, 120.0, 80.0, 10.0, 10.0, 10.0])
        R = np.diag([0.8, 0.8, 0.8])

        P = solve_continuous_are(A, B, Q, R)
        self.K = np.linalg.solve(R, B.T @ P)

        self.x = None
        self.rows = []
        self.t0 = time.time()
        self.csv_path = f"/tmp/fr3_leader_torque_{now_str()}.csv"

        self.sub = self.create_subscription(JointState, self.js_topic, self.cb_js, 10)
        self.pub = self.create_publisher(Float64MultiArray, self.cmd_topic, 10)
        self.timer = self.create_timer(1.0 / self.rate_hz, self.tick)

        self.get_logger().info(f"FR3 leader torque log CSV: {self.csv_path}")
        self.get_logger().info(f"joints={self.joints}")
        self.get_logger().info(f"sub={self.js_topic}")
        self.get_logger().info(f"pub={self.cmd_topic}")
        self.get_logger().info(f"q_des={self.q_des}")
        self.get_logger().info(f"K=\n{self.K}")

    def cb_js(self, msg):
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

    def tick(self):
        if self.x is None:
            return

        t = time.time() - self.t0
        if t >= self.duration_s:
            self.save_and_exit()
            return

        e = self.x - self.x_eq
        tau = self.u_eq - self.K @ e
        tau = np.clip(tau, -self.tau_limit, self.tau_limit)

        msg = Float64MultiArray()
        msg.data = [float(tau[0]), float(tau[1]), float(tau[2])]
        self.pub.publish(msg)

        self.rows.append([
            t,
            self.x[0], self.x[1], self.x[2],
            self.x[3], self.x[4], self.x[5],
            e[0], e[1], e[2], e[3], e[4], e[5],
            tau[0], tau[1], tau[2],
        ])

    def save_and_exit(self):
        with open(self.csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow([
                't',
                'q3','q4','q5',
                'dq3','dq4','dq5',
                'e3','e4','e5','de3','de4','de5',
                'tau3','tau4','tau5'
            ])
            w.writerows(self.rows)

        self.get_logger().info(f"Saved N={len(self.rows)} rows to {self.csv_path}")
        rclpy.shutdown()


def main():
    rclpy.init()
    node = FR3LeaderTorqueLog()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
