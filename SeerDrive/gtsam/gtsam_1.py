#!/usr/bin/env python3
"""
gtsam_1.py

Usage:
    python gtsam_1.py seerdrive_traj.npy

This script:
  - Loads a SeerDrive trajectory from a .npy file (dict → first trajectory).
  - Extracts (x, y) from (N, 3).
  - Builds a GTSAM factor graph:
        * prior on initial state
        * priors on each point
        * smoothness factors
  - Optimizes the trajectory
  - Saves optimized trajectory as <input>_opt.npy
  - Plots original vs optimized trajectories
  - Saves the plot as <input>_plot.png
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import gtsam
from gtsam import symbol


# -------------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------------

def load_trajectory_npy(path: str) -> np.ndarray:
    """
    Load a single trajectory from your SeerDrive .npy file.

    File structure:
        raw → ndarray (shape (), dtype object)
        inner → dict {token: np.ndarray(N,3)}

    Returns:
        traj_xy: (N,2) numpy array
    """
    raw = np.load(path, allow_pickle=True)

    if isinstance(raw, np.ndarray) and raw.shape == () and raw.dtype == object:
        inner = raw.item()
        keys = list(inner.keys())
        first_key = keys[0]
        traj = np.array(inner[first_key], dtype=float)

        print(f"Loaded inner dict with {len(inner)} trajectories.")
        print(f"Using key: {first_key}, trajectory shape: {traj.shape}")

    elif isinstance(raw, np.ndarray):
        traj = np.array(raw, dtype=float)
        print(f"Loaded numeric array directly, shape: {traj.shape}")

    else:
        raise ValueError(f"Unsupported .npy structure: outer type {type(raw)}")

    if traj.ndim != 2 or traj.shape[1] < 2:
        raise ValueError(f"Trajectory must be (N,2+) shaped, got {traj.shape}")

    return traj[:, :2]  # only x,y


def save_trajectory_npy(path: str, traj: np.ndarray) -> None:
    np.save(path, traj)


def build_noise(sigmas):
    sigmas = np.array(sigmas, dtype=float)
    return gtsam.noiseModel.Diagonal.Sigmas(sigmas)


# -------------------------------------------------------------------------
# Optimization
# -------------------------------------------------------------------------

def optimize_trajectory(
    seer_traj: np.ndarray,
    dt: float = 0.5,
    sigma_prior: float = 0.05,
    sigma_seer: float = 0.5,
    sigma_vel: float = 1.0,
    max_iters: int = 50,
) -> np.ndarray:

    N = seer_traj.shape[0]
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    prior_noise = build_noise([sigma_prior, sigma_prior])
    seer_noise = build_noise([sigma_seer, sigma_seer])
    vel_noise = build_noise([sigma_vel, sigma_vel])

    # Insert initial values
    for k, (x, y) in enumerate(seer_traj):
        initial.insert(symbol('x', k), gtsam.Point2(float(x), float(y)))

    # Prior on first state
    graph.add(gtsam.PriorFactorPoint2(
        symbol('x', 0),
        gtsam.Point2(float(seer_traj[0, 0]), float(seer_traj[0, 1])),
        prior_noise
    ))

    # Priors for each point to stay close
    for k, (x, y) in enumerate(seer_traj):
        graph.add(gtsam.PriorFactorPoint2(
            symbol('x', k),
            gtsam.Point2(float(x), float(y)),
            seer_noise
        ))

    # Smoothness
    zero = gtsam.Point2(0.0, 0.0)
    for k in range(N - 1):
        graph.add(gtsam.BetweenFactorPoint2(
            symbol('x', k), symbol('x', k + 1),
            zero,
            vel_noise
        ))

    # Optimize
    params = gtsam.GaussNewtonParams()
    params.setMaxIterations(max_iters)

    result = gtsam.GaussNewtonOptimizer(graph, initial, params).optimize()

    # Extract solution
    traj_opt = np.zeros_like(seer_traj)
    for k in range(N):
        pt = np.asarray(result.atPoint2(symbol('x', k))).ravel()
        traj_opt[k] = pt

    return traj_opt


# -------------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------------

def plot_trajectories_and_save(seer_traj, traj_opt, png_path, title=""):
    plt.figure(figsize=(6, 6))

    plt.plot(seer_traj[:, 0], seer_traj[:, 1],
             linestyle='--', marker='o', label='SeerDrive (original)')
    plt.plot(traj_opt[:, 0], traj_opt[:, 1],
             linestyle='-', marker='x', label='GTSAM optimized')

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(title if title else "SeerDrive vs GTSAM optimized")
    plt.legend()
    plt.axis('equal')
    plt.grid(True)

    # SAVE PNG
    plt.savefig(png_path, dpi=200, bbox_inches='tight')
    print(f"Saved plot as PNG: {png_path}")

    plt.show()


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python gtsam_1.py seerdrive_traj.npy")
        sys.exit(1)

    in_path = sys.argv[1]

    if not os.path.isfile(in_path):
        print(f"ERROR: File not found → {in_path}")
        sys.exit(1)

    # Load trajectory
    seer_traj = load_trajectory_npy(in_path)
    print(f"Loaded trajectory shape: {seer_traj.shape}")

    # Optimize
    traj_opt = optimize_trajectory(seer_traj)

    # Save optimized trajectory
    base, _ = os.path.splitext(in_path)
    out_npy = base + "_opt.npy"
    save_trajectory_npy(out_npy, traj_opt)
    print(f"Saved optimized trajectory: {out_npy}")

    # Save plot
    out_png = base + "_plot.png"
    plot_title = os.path.basename(in_path)

    plot_trajectories_and_save(seer_traj, traj_opt, out_png, title=plot_title)


if __name__ == "__main__":
    main()
