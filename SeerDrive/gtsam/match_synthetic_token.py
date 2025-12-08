#!/usr/bin/env python3
import sys
import csv
import numpy as np
from pathlib import Path

# ------------------------------------------------------------------
# Fixed paths
# ------------------------------------------------------------------
SEERDRIVE_TRAJ_PATH = (
    "/scratch/rob535f25s001_class_root/rob535f25s001_class/"
    "kushkp/SeerOpt/SeerDrive/gtsam/seerdrive_traj.npy"
)

SYNTHETIC_CSV_PATH = (
    "/scratch/rob535f25s001_class_root/rob535f25s001_class/"
    "kushkp/SeerOpt/SeerDrive/data/navsim/navhard_two_stage/"
    "synthetic_scenes_attributes.csv"
)


def load_pred_traj(token: str) -> np.ndarray:
    """Load SeerDrive predicted trajectory for a given token from fixed .npy dict."""
    npy_path = str(Path(SEERDRIVE_TRAJ_PATH).expanduser())
    data = np.load(npy_path, allow_pickle=True)
    inner = data.item()  # dict: key -> trajectory

    if token not in inner:
        sample_keys = list(inner.keys())[:10]
        raise KeyError(
            f"Token {token} not found in {npy_path}.\n"
            f"Example keys: {sample_keys}"
        )

    traj = np.asarray(inner[token])
    if traj.ndim != 2 or traj.shape[1] < 2:
        raise ValueError(f"Trajectory for {token} has unexpected shape: {traj.shape}")
    return traj  # (T, D>=2)


def parse_global_traj(global_traj_str: str) -> np.ndarray:
    """
    Parse the 'global_trajectory' CSV field, e.g.:

      [array([ 6.64429182e+05,  3.99840929e+06, -1.44118566e+00]),
       array([ ... ]),
       ...]

    into a numpy array of shape (N, 3).

    We use eval with a restricted environment where 'array' is np.array.
    """
    # Restricted environment: only allow 'array' mapped to np.array
    safe_globals = {"__builtins__": {}}
    safe_locals = {"array": np.array}

    try:
        obj = eval(global_traj_str, safe_globals, safe_locals)
    except Exception as e:
        raise ValueError(
            f"Failed to parse global_trajectory string:\n{global_traj_str}\nError: {e}"
        )

    # obj should be a list of np.arrays or something similar
    arr = np.array(obj, dtype=float)

    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Parsed global_trajectory has unexpected shape: {arr.shape}")
    return arr  # (N, D>=2)


def traj_alignment_error(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Simple alignment error between predicted traj and global_trajectory.
    Both arrays are (T, D>=2). Compare only x,y and truncate to min length.
    """
    pred_xy = pred[:, :2]
    gt_xy = gt[:, :2]

    min_len = min(len(pred_xy), len(gt_xy))
    if min_len == 0:
        return float("inf")

    pred_xy = pred_xy[:min_len]
    gt_xy = gt_xy[:min_len]

    diff = pred_xy - gt_xy
    mse = np.mean(np.sum(diff ** 2, axis=1))  # mean squared Euclidean distance
    return float(mse)


def main():
    if len(sys.argv) != 2:
        print("Usage:")
        print("  python match_synthetic_token.py ORIGINAL_TOKEN")
        print("\nPaths are fixed inside the script:")
        print(f"  seerdrive_traj.npy: {SEERDRIVE_TRAJ_PATH}")
        print(f"  synthetic_scenes_attributes.csv: {SYNTHETIC_CSV_PATH}")
        sys.exit(1)

    orig_token = sys.argv[1]

    npy_path = str(Path(SEERDRIVE_TRAJ_PATH).expanduser())
    csv_path = str(Path(SYNTHETIC_CSV_PATH).expanduser())

    print(f"Loading predicted trajectory from: {npy_path}")
    print(f"Original token (corresponding_original_scene_token): {orig_token}")
    print(f"CSV: {csv_path}\n")

    # 1) Load predicted trajectory
    pred_traj = load_pred_traj(orig_token)
    print(f"Predicted traj shape for {orig_token}: {pred_traj.shape}\n")

    # 2) Scan CSV rows and compute error for matches
    best_row = None
    best_err = float("inf")
    candidates = []

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("corresponding_original_scene_token") != orig_token:
                continue

            syn_token = row.get("synthetic_scene_token", "")
            gtraj_str = row.get("global_trajectory", "")

            if not gtraj_str:
                continue

            try:
                gt_traj = parse_global_traj(gtraj_str)
                err = traj_alignment_error(pred_traj, gt_traj)
            except Exception as e:
                print(
                    f"[WARN] Failed to parse/compare row "
                    f"synthetic_scene_token={syn_token}: {e}"
                )
                continue

            candidates.append((syn_token, err))
            if err < best_err:
                best_err = err
                best_row = (syn_token, row, gt_traj)

    if not candidates:
        print(f"No CSV rows found with corresponding_original_scene_token == {orig_token}")
        sys.exit(1)

    # 3) Print all candidates sorted by error
    candidates_sorted = sorted(candidates, key=lambda x: x[1])
    print("Candidate synthetic_scene_tokens for this original token, sorted by error:")
    for syn_token, err in candidates_sorted:
        print(f"  synthetic_scene_token={syn_token}   alignment_error={err:.4f}")

    # 4) Best match
    best_syn_token, best_row_dict, best_gt = best_row
    print("\n===== BEST MATCH =====")
    print(f"Best synthetic_scene_token: {best_syn_token}")
    print(f"Alignment error (MSE in XY): {best_err:.6f}")
    print("\nRow summary:")
    print(f"  log_name: {best_row_dict.get('log_name')}")
    print(f"  viewpoint: {best_row_dict.get('viewpoint')}")
    print(f"  map_name: {best_row_dict.get('map_name')}")
    print(f"  pair_identifier: {best_row_dict.get('pair_identifier')}")
    print(f"  global_trajectory length: {best_gt.shape[0]} points")

    print("\nDone.")


if __name__ == "__main__":
    main()
