#!/usr/bin/env python3
import os
import csv
import argparse
import numpy as np


def read_csv_matrix(path):
    with open(path, "r") as f:
        r = csv.reader(f)
        next(r, None)
        data = [[float(v) for v in row] for row in r if row]
    return np.asarray(data, dtype=float)


def save_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def build_phi(Ek, Uk, Ek1, Uk1):
    N = Ek.shape[0]
    n = Ek.shape[1]
    m = Uk.shape[1]
    d = n + m

    Phi = np.zeros((N, d * d), dtype=float)

    for k in range(N):
        zk = np.concatenate([Ek[k], Uk[k]])
        zk1 = np.concatenate([Ek1[k], Uk1[k]])
        Phi[k, :] = np.kron(zk, zk) - np.kron(zk1, zk1)

    return Phi


def eval_K_from_Qdiag(q_diag, Phi, Ek, Uk, R, reg_huu):
    N, n = Ek.shape
    m = Uk.shape[1]

    Q = np.diag(q_diag)

    theta = np.zeros(N, dtype=float)
    for k in range(N):
        theta[k] = Ek[k] @ Q @ Ek[k] + Uk[k] @ R @ Uk[k]

    vecH, *_ = np.linalg.lstsq(Phi, theta, rcond=None)

    H = vecH.reshape(n + m, n + m)
    H = 0.5 * (H + H.T)

    Hux = H[n:, :n]
    Huu = 0.5 * (H[n:, n:] + H[n:, n:].T)
    Huu = Huu + reg_huu * np.eye(m)

    K = np.linalg.solve(Huu, Hux)
    return K


def objective(q_diag, Phi, Ek, Uk, R, K_star, q_ref, beta_Q, reg_huu):
    K = eval_K_from_Qdiag(q_diag, Phi, Ek, Uk, R, reg_huu)

    Jk = float(np.linalg.norm(K - K_star, "fro") ** 2)

    q_ratio = q_diag / q_ref
    Jq = float(np.linalg.norm(q_ratio - 1.0) ** 2)

    J = Jk + beta_Q * Jq
    return J, K, Jk, Jq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/tmp/fr3_freemodel_leader")
    ap.add_argument("--out_dir", default="/tmp/fr3_freemodel_learned")

    ap.add_argument("--maxIter", type=int, default=1500)
    ap.add_argument("--tol_K", type=float, default=1e-3)
    ap.add_argument("--alphaQ", type=float, default=0.2)
    ap.add_argument("--c_spsa", type=float, default=1e-5)
    ap.add_argument("--pd_floor", type=float, default=1e-6)
    ap.add_argument("--reg_huu", type=float, default=1e-4)
    ap.add_argument("--R_diag", type=float, nargs="+", default=[0.8, 0.8, 0.8])

    # Q-shape regularization.
    # This should match the leader forward Q_star.
    ap.add_argument("--Q_ref_diag", type=float, nargs="+",
                    default=[100.0, 100.0, 100.0, 10.0, 10.0, 10.0])

    # Initial SPSA guess. This is separate from Q_ref_diag.
    ap.add_argument("--Q_init_diag", type=float, nargs="+",
                    default=[10.0, 10.0, 10.0, 10.0, 10.0, 10.0])

    ap.add_argument("--beta_Q", type=float, default=10.0)

    args = ap.parse_args()

    Ek = read_csv_matrix(os.path.join(args.data_dir, "Ek.csv"))
    Uk = read_csv_matrix(os.path.join(args.data_dir, "Uk.csv"))
    Ek1 = read_csv_matrix(os.path.join(args.data_dir, "Ek1.csv"))
    Uk1 = read_csv_matrix(os.path.join(args.data_dir, "Uk1.csv"))
    K_star = read_csv_matrix(os.path.join(args.data_dir, "K_star.csv"))

    N, n = Ek.shape
    m = Uk.shape[1]

    R = np.diag(np.asarray(args.R_diag, dtype=float))
    q_ref = np.asarray(args.Q_ref_diag, dtype=float)
    q_init = np.asarray(args.Q_init_diag, dtype=float)

    if q_ref.shape[0] != n:
        raise RuntimeError(f"Q_ref_diag must have length {n}, got {q_ref.shape[0]}")
    if q_init.shape[0] != n:
        raise RuntimeError(f"Q_init_diag must have length {n}, got {q_init.shape[0]}")

    print("========== FR3 freemodel offline SPSA ==========")
    print(f"N={N}, n={n}, m={m}")
    print(f"data_dir={args.data_dir}")
    print(f"out_dir={args.out_dir}")
    print(f"maxIter={args.maxIter}, alphaQ={args.alphaQ}, beta_Q={args.beta_Q}")
    print(f"Q_ref_diag={q_ref}")
    print(f"Q_init_diag={q_init}")

    Phi = build_phi(Ek, Uk, Ek1, Uk1)
    rank_phi = np.linalg.matrix_rank(Phi)
    print(f"rank(Phi)={rank_phi}/{Phi.shape[1]}")

    # Start from the chosen initial diagonal, not from the reference.
    q_diag = q_init.copy()

    hist = []
    final_err = None
    final_Jq = None
    converged = False

    for it in range(args.maxIter):
        J, K_now, Jk, Jq = objective(
            q_diag, Phi, Ek, Uk, R, K_star, q_ref, args.beta_Q, args.reg_huu
        )

        kerr = float(np.linalg.norm(K_now - K_star, "fro"))
        final_err = kerr
        final_Jq = Jq

        hist.append([it, kerr, J, Jk, Jq, *q_diag.tolist()])

        if it < 10 or it % 50 == 0:
            print(
                f"iter={it:04d}, "
                f"||K-K*||={kerr:.6e}, "
                f"Jq={Jq:.6e}, "
                f"Qdiag={np.round(q_diag, 4)}"
            )

        if kerr <= args.tol_K:
            converged = True
            print(f"converged at iter={it}, err={kerr:.6e}")
            break

        Delta = 2.0 * (np.random.rand(n) > 0.5).astype(float) - 1.0
        c = args.c_spsa

        q_plus = np.maximum(q_diag + c * Delta, args.pd_floor)
        q_minus = np.maximum(q_diag - c * Delta, args.pd_floor)

        J_plus, _, _, _ = objective(
            q_plus, Phi, Ek, Uk, R, K_star, q_ref, args.beta_Q, args.reg_huu
        )
        J_minus, _, _, _ = objective(
            q_minus, Phi, Ek, Uk, R, K_star, q_ref, args.beta_Q, args.reg_huu
        )

        ghat = (J_plus - J_minus) / (2.0 * c * Delta)
        q_diag = np.maximum(q_diag - args.alphaQ * ghat, args.pd_floor)

    K_learned = eval_K_from_Qdiag(q_diag, Phi, Ek, Uk, R, args.reg_huu)

    os.makedirs(args.out_dir, exist_ok=True)

    np.save(os.path.join(args.out_dir, "Q_learned.npy"), np.diag(q_diag))
    np.save(os.path.join(args.out_dir, "K_learned.npy"), K_learned)

    save_csv(
        os.path.join(args.out_dir, "Q_learned.csv"),
        ["index", "value"],
        [[i, float(v)] for i, v in enumerate(q_diag)],
    )
    save_csv(
        os.path.join(args.out_dir, "K_learned.csv"),
        ["k1", "k2", "k3", "k4", "k5", "k6"],
        [[float(v) for v in row] for row in K_learned],
    )
    save_csv(
        os.path.join(args.out_dir, "spsa_history.csv"),
        ["it", "K_err", "J", "Jk", "Jq", "q1", "q2", "q3", "q4", "q5", "q6"],
        hist,
    )
    save_csv(
        os.path.join(args.out_dir, "result_summary.csv"),
        ["metric", "value"],
        [
            ["rank_Phi", int(rank_phi)],
            ["converged", int(converged)],
            ["final_K_err", float(final_err)],
            ["final_Jq", float(final_Jq)],
        ],
    )

    print("========== RESULT ==========")
    print("Q_ref_diag =", np.round(q_ref, 6))
    print("Q_learned_diag =", np.round(q_diag, 6))
    print("K_learned =")
    print(np.round(K_learned, 6))
    print("K_star =")
    print(np.round(K_star, 6))
    print("||K_learned-K_star||_F =", np.linalg.norm(K_learned - K_star, "fro"))
    print("saved to", args.out_dir)


if __name__ == "__main__":
    main()
