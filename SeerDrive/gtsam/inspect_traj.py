#!/usr/bin/env python3
"""
inspect_traj.py

Usage:
    # 1) Inspect and plot ALL trajectories
    python inspect_traj.py seerdrive_traj.npy

    # 2) Inspect and plot ONLY a specific trajectory (by key/token)
    python inspect_traj.py seerdrive_traj.npy <key>

Behavior:
  - Always prints inner dict type and number of trajectories.
  - If no key is given:
        * prints key/type/shape for ALL trajectories
        * plots up to the first 1000 trajectories on one figure
        * saves: <base>_seerdrive_traj.png
          (e.g. seerdrive_traj_seerdrive_traj.png)
  - If a key is given:
        * prints key/type/shape only for that trajectory
        * plots only that trajectory
        * saves: <base>_<key>.png
          (e.g. seerdrive_traj_0a3ddcb280eb502a.png)
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python inspect_traj.py seerdrive_traj.npy")
        print("  python inspect_traj.py seerdrive_traj.npy <key>")
        sys.exit(1)

    path = sys.argv[1]
    key_filter = sys.argv[2] if len(sys.argv) >= 3 else None

    data = np.load(path, allow_pickle=True)

    # unwrap the inner dict
    inner = data.item()
    print(f"Inner type: {type(inner)}")
    print(f"Number of trajectories: {len(inner)}\n")

    base, _ = os.path.splitext(os.path.basename(path))

    # -----------------------------
    # MODE 2: specific key/token
    # -----------------------------
    if key_filter is not None:
        if key_filter not in inner:
            print(f"ERROR: key '{key_filter}' not found in prediction dict.")
            print("Available keys (first 10):")
            for i, k in enumerate(inner.keys()):
                if i >= 10:
                    break
                print("  ", k)
            sys.exit(1)

        v = inner[key_filter]
        arr = np.asarray(v)

        print("Inspecting ONLY key:", key_filter)
        print(f"type = {type(v)}, shape = {getattr(arr, 'shape', 'no shape')}")

        # Plot this single trajectory if it looks like (N, >=2)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            x = arr[:, 0]
            y = arr[:, 1]

            plt.figure(figsize=(6, 6))
            plt.plot(x, y, marker="o")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title(f"SeerDrive trajectory for key = {key_filter}")
            plt.axis("equal")
            plt.grid(True)

            out_png = f"{base}_{key_filter}.png"
            plt.savefig(out_png, dpi=200, bbox_inches="tight")
            print(f"\nSaved SeerDrive trajectory plot for key '{key_filter}' → {out_png}\n")
            plt.show()
        else:
            print("\nTrajectory for this key is not a 2D array with at least 2 columns. No plot created.\n")

        return  # done in specific-key mode

    # -----------------------------
    # MODE 1: all trajectories
    # -----------------------------
    print("All trajectories and their shapes:")

    xs_list = []
    ys_list = []

    for i, (k, v) in enumerate(inner.items()):
        arr = np.asarray(v)
        print(f"{i}: key = {k}, type = {type(v)}, shape = {getattr(arr, 'shape', 'no shape')}")

        # collect for plotting if it looks like (N, >=2)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            if len(xs_list) < 1000:  # just to avoid over-plotting millions
                xs_list.append(arr[:, 0])
                ys_list.append(arr[:, 1])

    # ---- Plot and save ----
    if xs_list:
        plt.figure(figsize=(6, 6))
        for x, y in zip(xs_list, ys_list):
            plt.plot(x, y, alpha=0.3)  # light lines for many trajectories

        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("SeerDrive trajectories (first up to 1000)")
        plt.axis("equal")
        plt.grid(True)

        out_png = base + "_seerdrive_traj.png"
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
        print(f"\nSaved SeerDrive trajectory plot → {out_png}\n")
        plt.show()
    else:
        print("\nNo valid (N,2+) trajectories found to plot.\n")


if __name__ == "__main__":
    main()
