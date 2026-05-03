#!/usr/bin/env python3
import argparse
import csv
import os
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', default='/tmp/fr3_torque_K.npz')
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(args.csv)

    X = []
    U = []

    with open(args.csv, 'r') as f:
        r = csv.DictReader(f)
        need = ['e3','e4','e5','de3','de4','de5','tau3','tau4','tau5']
        for k in need:
            if k not in r.fieldnames:
                raise RuntimeError(f"CSV missing {k}; got {r.fieldnames}")

        for row in r:
            x = [
                float(row['e3']),
                float(row['e4']),
                float(row['e5']),
                float(row['de3']),
                float(row['de4']),
                float(row['de5']),
            ]
            u = [
                float(row['tau3']),
                float(row['tau4']),
                float(row['tau5']),
            ]
            X.append(x)
            U.append(u)

    X = np.asarray(X, dtype=float)   # N x 6
    U = np.asarray(U, dtype=float)   # N x 3

    print(f"Loaded N={X.shape[0]} samples")
    if X.shape[0] < 20:
        raise RuntimeError("Too few samples. Increase duration_s.")

    # Solve U ≈ -X K^T.
    # np.linalg.lstsq gives A where X @ A ≈ U, so K = -A.T.
    A, residuals, rank, svals = np.linalg.lstsq(X, U, rcond=None)
    K = -A.T

    print("rank(X) =", rank)
    print("singular values =", svals)
    print("K_hat 3x6 =")
    print(K)

    np.savez(args.out, K=K, csv=args.csv, N=X.shape[0], rank=rank, svals=svals)
    print("Saved:", args.out)


if __name__ == '__main__':
    main()
