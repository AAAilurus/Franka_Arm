#!/usr/bin/env python3
import os
import csv
import numpy as np
import scipy.linalg

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from iocfranka_3dof.fr3_j345_exact_lqr import p, arm_f, linearize_numerical


def save_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def make_joint_names(prefix):
    if prefix.endswith("fr3_"):
        return [f"{prefix}joint3", f"{prefix}joint4", f"{prefix}joint5"]
    return [f"{prefix}fr3_joint3", f"{prefix}fr3_joint4", f"{prefix}fr3_joint5"]


def discretize_zoh(A, B, dt):
    n = A.shape[0]
    m = B.shape[1]

    M = np.zeros((n + m, n + m), dtype=float)
    M[:n, :n] = A
    M[:n, n:n+m] = B

    Md = scipy.linalg.expm(M * dt)
    Ad = Md[:n, :n]
    Bd = Md[:n, n:n+m]
    return Ad, Bd


def compute_lqr_gain(q_eq, dt):
    x_eq = np.hstack((q_eq, np.zeros(3)))
    u_eq = np.zeros(3)

    A, B = linearize_numerical(lambda x, u: arm_f(x, u, p), x_eq, u_eq)

    Q_star = np.diag([100.0, 100.0, 100.0, 10.0, 10.0, 10.0])
    R_star = np.diag([0.8, 0.8, 0.8])

    P = scipy.linalg.solve_continuous_are(A, B, Q_star, R_star)
    K = np.linalg.solve(R_star, B.T @ P)

    Ad, Bd = discretize_zoh(A, B, dt)

    return K, x_eq, u_eq, A, B, Ad, Bd


class FR3FMLeaderCollect(Node):
    """
    Multi-trajectory FR3 freemodel data collector.

    For each trajectory:
      1. Drive leader to a random start pose.
      2. Switch desired pose to final target.
      3. Record samples_per_traj transitions while exact LQR moves toward target.

    Saved data:
      Ek.csv, Uk.csv, Ek1.csv, Uk1.csv, K_star.csv
    """

    def __init__(self):
        super().__init__("fr3_fm_leader_collect")

        self.declare_parameter("joint_name_prefix", "leader_fr3_")
        self.declare_parameter("joint_states_topic", "/leader/joint_states")
        self.declare_parameter("command_topic", "/leader/arm_3dof_cmd/commands")

        # final target
        self.declare_parameter("q_des_j3", 0.3)
        self.declare_parameter("q_des_j4", -1.8)
        self.declare_parameter("q_des_j5", 0.2)

        self.declare_parameter("rate_hz", 200.0)

        # multi-trajectory setting
        self.declare_parameter("n_traj", 50)
        self.declare_parameter("samples_per_traj", 300)
        self.declare_parameter("reset_steps", 500)

        # random start region around the standing pose [0, -2.36, 0]
        self.declare_parameter("start_j3_min", -0.35)
        self.declare_parameter("start_j3_max", 0.35)
        self.declare_parameter("start_j4_min", -2.65)
        self.declare_parameter("start_j4_max", -2.05)
        self.declare_parameter("start_j5_min", -0.70)
        self.declare_parameter("start_j5_max", 0.70)

        self.declare_parameter("noise_std", 0.02)
        self.declare_parameter("noise_std_j3", 0.04)
        self.declare_parameter("noise_std_j4", 0.04)
        self.declare_parameter("noise_std_j5", 0.08)
        self.declare_parameter("tau_limit", 5.0)
        self.declare_parameter("sat_margin", 0.05)

        # If true, save SO100-style model-predicted next transition:
        #   e_next = Ad e + Bd u
        # instead of measured next joint_state.
        self.declare_parameter("use_model_next", True)

        self.declare_parameter("save_dir", "/tmp/fr3_freemodel_leader")

        prefix = str(self.get_parameter("joint_name_prefix").value)
        self.joints = make_joint_names(prefix)

        self.joint_states_topic = str(self.get_parameter("joint_states_topic").value)
        self.command_topic = str(self.get_parameter("command_topic").value)

        self.q_final = np.array([
            float(self.get_parameter("q_des_j3").value),
            float(self.get_parameter("q_des_j4").value),
            float(self.get_parameter("q_des_j5").value),
        ], dtype=float)

        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.dt = 1.0 / self.rate_hz

        self.n_traj = int(self.get_parameter("n_traj").value)
        self.samples_per_traj = int(self.get_parameter("samples_per_traj").value)
        self.reset_steps = int(self.get_parameter("reset_steps").value)

        self.start_min = np.array([
            float(self.get_parameter("start_j3_min").value),
            float(self.get_parameter("start_j4_min").value),
            float(self.get_parameter("start_j5_min").value),
        ], dtype=float)

        self.start_max = np.array([
            float(self.get_parameter("start_j3_max").value),
            float(self.get_parameter("start_j4_max").value),
            float(self.get_parameter("start_j5_max").value),
        ], dtype=float)

        self.noise_std = float(self.get_parameter("noise_std").value)
        self.noise_vec = np.array([
            float(self.get_parameter("noise_std_j3").value),
            float(self.get_parameter("noise_std_j4").value),
            float(self.get_parameter("noise_std_j5").value),
        ], dtype=float)
        self.tau_limit = float(self.get_parameter("tau_limit").value)
        self.sat_margin = float(self.get_parameter("sat_margin").value)
        self.use_model_next = bool(self.get_parameter("use_model_next").value)
        self.save_dir = str(self.get_parameter("save_dir").value)

        # Expert final-target gain used for recording and K_star.
        self.K_star, self.x_eq_final, self.u_eq, self.A_final, self.B_final, self.Ad_final, self.Bd_final = compute_lqr_gain(self.q_final, self.dt)

        self.x = None

        self.traj_id = 0
        self.step_in_phase = 0
        self.phase = "reset"

        self.q_start = self.sample_start()
        self.K_reset, self.x_eq_reset, self.u_eq_reset, self.A_reset, self.B_reset, self.Ad_reset, self.Bd_reset = compute_lqr_gain(self.q_start, self.dt)

        self.prev_e = None
        self.prev_u = None
        self.prev_valid = False
        self.skipped_saturated = 0

        self.Ek = []
        self.Uk = []
        self.Ek1 = []
        self.Uk1 = []
        self.meta = []

        self.sub = self.create_subscription(JointState, self.joint_states_topic, self.cb_js, 10)
        self.pub = self.create_publisher(Float64MultiArray, self.command_topic, 10)
        self.timer = self.create_timer(self.dt, self.step)

        self.get_logger().info("========== FR3 freemodel leader collect: multi-trajectory ==========")
        self.get_logger().info(f"joints={self.joints}")
        self.get_logger().info(f"sub={self.joint_states_topic}")
        self.get_logger().info(f"pub={self.command_topic}")
        self.get_logger().info(f"final target q={self.q_final}")
        self.get_logger().info(f"n_traj={self.n_traj}, samples_per_traj={self.samples_per_traj}, total={self.n_traj*self.samples_per_traj}")
        self.get_logger().info(f"start_min={self.start_min}, start_max={self.start_max}")
        self.get_logger().info(f"noise_vec={self.noise_vec}")
        self.get_logger().info(f"tau_limit={self.tau_limit}, sat_margin={self.sat_margin}")
        self.get_logger().info(f"use_model_next={self.use_model_next}")
        self.get_logger().info(f"Ad_final=\n{np.round(self.Ad_final, 5)}")
        self.get_logger().info(f"Bd_final=\n{np.round(self.Bd_final, 5)}")
        self.get_logger().info(f"save_dir={self.save_dir}")
        self.get_logger().info(f"K_star=\n{np.round(self.K_star, 4)}")

    def sample_start(self):
        return self.start_min + (self.start_max - self.start_min) * np.random.rand(3)

    def cb_js(self, msg):
        try:
            idx = [msg.name.index(j) for j in self.joints]
        except ValueError:
            return

        q = np.array([msg.position[i] for i in idx], dtype=float)
        dq = np.array([msg.velocity[i] if len(msg.velocity) > i else 0.0 for i in idx], dtype=float)
        self.x = np.hstack((q, dq))

    def publish_tau(self, tau):
        msg = Float64MultiArray()
        msg.data = [float(tau[0]), float(tau[1]), float(tau[2])]
        self.pub.publish(msg)

    def exact_lqr_tau(self, K, x_eq, u_eq, noise=True):
        tau_raw = u_eq - K @ (self.x - x_eq)

        if noise:
            tau_raw = tau_raw + self.noise_vec * np.random.randn(3)

        saturated = bool(np.any(np.abs(tau_raw) >= (self.tau_limit - self.sat_margin)))
        tau = np.clip(tau_raw, -self.tau_limit, self.tau_limit)
        return tau, saturated

    def start_next_traj(self):
        self.traj_id += 1

        if self.traj_id >= self.n_traj:
            self.save_all()
            rclpy.shutdown()
            return

        self.phase = "reset"
        self.step_in_phase = 0
        self.prev_e = None
        self.prev_u = None
        self.prev_valid = False

        self.q_start = self.sample_start()
        self.K_reset, self.x_eq_reset, self.u_eq_reset, self.A_reset, self.B_reset, self.Ad_reset, self.Bd_reset = compute_lqr_gain(self.q_start, self.dt)

        self.get_logger().info(
            f"starting traj {self.traj_id+1}/{self.n_traj}, q_start={np.round(self.q_start, 4)}"
        )

    def step(self):
        if self.x is None:
            return

        q = self.x[0:3]
        dq = self.x[3:6]

        if self.phase == "reset":
            # Drive to random start pose. Do not record during reset.
            tau, _ = self.exact_lqr_tau(self.K_reset, self.x_eq_reset, self.u_eq_reset, noise=False)
            self.publish_tau(tau)

            self.step_in_phase += 1

            if self.step_in_phase >= self.reset_steps:
                self.phase = "record"
                self.step_in_phase = 0
                self.prev_e = None
                self.prev_u = None
                self.prev_valid = False
                self.get_logger().info(
                    f"record traj {self.traj_id+1}/{self.n_traj}: "
                    f"q_start_actual={np.round(q, 4)}, dq={np.round(dq, 4)}"
                )

            return

        # Record phase: exact expert drives to final target.
        e = self.x - self.x_eq_final
        u, saturated = self.exact_lqr_tau(self.K_star, self.x_eq_final, self.u_eq, noise=True)
        self.publish_tau(u)

        current_valid = not saturated

        if self.use_model_next:
            # SO100-style model transition:
            #   e_next = Ad e + Bd u
            #   u_next = -K_star e_next + noise
            e_next = self.Ad_final @ e + self.Bd_final @ u

            u_next_raw = self.u_eq - self.K_star @ e_next
            if self.noise_vec is not None:
                u_next_raw = u_next_raw + self.noise_vec * np.random.randn(3)

            next_saturated = bool(np.any(np.abs(u_next_raw) >= (self.tau_limit - self.sat_margin)))
            u_next = np.clip(u_next_raw, -self.tau_limit, self.tau_limit)

            if current_valid and (not next_saturated):
                self.Ek.append(e.copy())
                self.Uk.append(u.copy())
                self.Ek1.append(e_next.copy())
                self.Uk1.append(u_next.copy())
                self.meta.append([self.traj_id, self.step_in_phase])
            else:
                self.skipped_saturated += 1

            # No measured-next buffering needed in model-next mode.
            self.prev_e = None
            self.prev_u = None
            self.prev_valid = False

        else:
            # Old measured-next transition:
            #   Ek  = previous measured e
            #   Ek1 = current measured e
            if self.prev_e is not None and self.prev_u is not None and self.prev_valid and current_valid:
                self.Ek.append(self.prev_e.copy())
                self.Uk.append(self.prev_u.copy())
                self.Ek1.append(e.copy())
                self.Uk1.append(u.copy())
                self.meta.append([self.traj_id, self.step_in_phase])
            else:
                if saturated:
                    self.skipped_saturated += 1

            self.prev_e = e.copy()
            self.prev_u = u.copy()
            self.prev_valid = current_valid

        self.step_in_phase += 1

        total = len(self.Ek)
        if total > 0 and total % 300 == 0:
            self.get_logger().info(
                f"saved samples={total}/{self.n_traj*self.samples_per_traj}, "
                f"traj={self.traj_id+1}/{self.n_traj}, "
                f"q={np.round(q, 4)}, dq={np.round(dq, 4)}, "
                f"|e|={np.linalg.norm(e):.4f}, u={np.round(u, 3)}"
            )

        if self.step_in_phase >= self.samples_per_traj:
            self.start_next_traj()

    def save_all(self):
        os.makedirs(self.save_dir, exist_ok=True)

        save_csv(
            os.path.join(self.save_dir, "Ek.csv"),
            ["e1", "e2", "e3", "e4", "e5", "e6"],
            self.Ek,
        )
        save_csv(
            os.path.join(self.save_dir, "Uk.csv"),
            ["u1", "u2", "u3"],
            self.Uk,
        )
        save_csv(
            os.path.join(self.save_dir, "Ek1.csv"),
            ["e1", "e2", "e3", "e4", "e5", "e6"],
            self.Ek1,
        )
        save_csv(
            os.path.join(self.save_dir, "Uk1.csv"),
            ["u1", "u2", "u3"],
            self.Uk1,
        )
        save_csv(
            os.path.join(self.save_dir, "K_star.csv"),
            ["k1", "k2", "k3", "k4", "k5", "k6"],
            self.K_star,
        )
        save_csv(
            os.path.join(self.save_dir, "meta.csv"),
            ["traj_id", "sample_id"],
            self.meta,
        )

        np.save(os.path.join(self.save_dir, "K_star.npy"), self.K_star)

        self.get_logger().info(
            f"saved {len(self.Ek)} samples from {self.n_traj} trajectories to {self.save_dir}; "
            f"skipped_saturated={self.skipped_saturated}"
        )


def main():
    rclpy.init()
    node = FR3FMLeaderCollect()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.save_all()
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


if __name__ == "__main__":
    main()
