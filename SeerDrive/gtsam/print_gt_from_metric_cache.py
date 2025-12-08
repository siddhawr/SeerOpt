#!/usr/bin/env python3
"""
print_gt_from_metric_cache.py

Usage:
    # Just GT / centerline / drivable area:
    python print_gt_from_metric_cache.py path/to/metric_cache.pkl

    # GT + SeerDrive prediction for a specific token/key:
    python print_gt_from_metric_cache.py path/to/metric_cache.pkl <seerdrive_key>

The script:
  - Loads a NAVSIM MetricCache.
  - Extracts:
      * drivable_area_map (polygons)
      * centerline (PDMPath)
      * ground-truth ego trajectory (InterpolatedTrajectory)
  - Optionally loads SeerDrive predicted trajectory from a fixed .npy file
    and transforms it from ego-local frame into the global map frame.
  - Trims the centerline to match the GT trajectory span.
  - Plots everything together and saves a PNG.
"""

import sys
import os
import lzma
import pickle
import numpy as np
import matplotlib.pyplot as plt

from shapely.geometry import Polygon

from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.evaluate.pdm_score import get_trajectory_as_array
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

# -------------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------------

SEERDRIVE_TRAJ_NPY = (
    "/scratch/rob535f25s001_class_root/rob535f25s001_class/"
    "kushkp/SeerOpt/SeerDrive/gtsam/seerdrive_traj.npy"
)


# -------------------------------------------------------------------------
# Loading utilities
# -------------------------------------------------------------------------

def load_metric_cache(path: str) -> MetricCache:
    """Try LZMA first, fallback to plain pickle."""
    try:
        with lzma.open(path, "rb") as f:
            return pickle.load(f)
    except lzma.LZMAError:
        with open(path, "rb") as f:
            return pickle.load(f)


# -------------------------------------------------------------------------
# Extraction helpers
# -------------------------------------------------------------------------

def extract_drivable_polygons(metric_cache: MetricCache):
    """
    Extract shapely Polygons from metric_cache.drivable_area_map.

    metric_cache.drivable_area_map is a PDMDrivableMap that has
    an internal _geometries list of shapely Polygons.
    """
    if not hasattr(metric_cache, "drivable_area_map"):
        raise RuntimeError("MetricCache has no attribute 'drivable_area_map'.")

    dam = metric_cache.drivable_area_map
    geoms = getattr(dam, "_geometries", None)
    if geoms is None:
        raise RuntimeError("drivable_area_map has no '_geometries' field.")

    polygons = [g for g in geoms if isinstance(g, Polygon)]
    if len(polygons) == 0:
        raise RuntimeError("No Polygon geometries found in drivable_area_map._geometries")

    return polygons


def extract_centerline_xy(metric_cache: MetricCache):
    """
    Extract centerline coordinates from metric_cache.centerline.

    centerline is a PDMPath with either:
      - .linestring (shapely LineString), or
      - ._states_se2_array (Nx3) -> [x, y, heading].
    """
    if not hasattr(metric_cache, "centerline"):
        raise RuntimeError("MetricCache has no attribute 'centerline'.")

    cl = metric_cache.centerline

    # Prefer linestring if present
    ls = getattr(cl, "linestring", None)
    if ls is not None:
        x, y = ls.xy
        return np.asarray(x), np.asarray(y)

    # Fallback: _states_se2_array
    arr = getattr(cl, "_states_se2_array", None)
    if arr is None:
        raise RuntimeError("centerline has neither 'linestring' nor '_states_se2_array'.")

    arr = np.asarray(arr)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise RuntimeError(f"Unexpected _states_se2_array shape: {arr.shape}")

    return arr[:, 0], arr[:, 1]


def sample_trajectory_xy(metric_cache: MetricCache):
    """
    Sample the ground-truth ego trajectory from metric_cache.trajectory.

    Uses:
      - start_time = metric_cache.ego_state.time_point
      - horizon = trajectory.duration (full duration, or 4s fallback)
      - interval_length = 0.5 s
    """
    if not hasattr(metric_cache, "trajectory"):
        raise RuntimeError("MetricCache has no attribute 'trajectory'.")

    traj = metric_cache.trajectory

    horizon = getattr(traj, "duration", None)
    if horizon is None:
        horizon = 4.0  # safe fallback

    start_time = metric_cache.ego_state.time_point
    future_sampling = TrajectorySampling(time_horizon=horizon, interval_length=0.5)

    traj_arr = get_trajectory_as_array(
        trajectory=traj,
        future_sampling=future_sampling,
        start_time=start_time,
    )

    if traj_arr.ndim != 2 or traj_arr.shape[1] < 2:
        raise RuntimeError(f"Sampled trajectory has unexpected shape: {traj_arr.shape}")

    return traj_arr[:, 0], traj_arr[:, 1]


def get_ego_pose_xyh(metric_cache: MetricCache):
    """
    Get ego pose (x, y, yaw) at the start from the MetricCache's ego_state.
    """
    if not hasattr(metric_cache, "ego_state"):
        raise RuntimeError("MetricCache has no attribute 'ego_state'.")

    ego = metric_cache.ego_state

    # Typical NuPlan EgoState: ego.rear_axle.x/y/heading
    if hasattr(ego, "rear_axle"):
        ra = ego.rear_axle
        return float(ra.x), float(ra.y), float(ra.heading)

    # Fallback: ego.center.x/y/heading
    if hasattr(ego, "center"):
        c = ego.center
        x0 = getattr(c, "x", None)
        y0 = getattr(c, "y", None)
        yaw0 = getattr(c, "heading", None)
        if x0 is not None and y0 is not None and yaw0 is not None:
            return float(x0), float(y0), float(yaw0)

    raise RuntimeError("Unable to extract ego pose (x,y,yaw) from ego_state.")


def local_to_global(x_local, y_local, x0, y0, yaw0):
    """
    Convert local (x_local, y_local) in ego frame at time 0
    to global (X, Y) using origin pose (x0, y0, yaw0).
    """
    x_local = np.asarray(x_local)
    y_local = np.asarray(y_local)

    c = np.cos(yaw0)
    s = np.sin(yaw0)

    X = x0 + c * x_local - s * y_local
    Y = y0 + s * x_local + c * y_local
    return X, Y


def load_seerdrive_trajectory(key: str):
    """
    Load SeerDrive predicted trajectory for a given key/token from the fixed .npy file.

    Assumes each trajectory has shape (N,3) = [t, x_local, y_local].
    """
    print(f"\n🔹 Loading SeerDrive trajectories from:\n{SEERDRIVE_TRAJ_NPY}")
    data = np.load(SEERDRIVE_TRAJ_NPY, allow_pickle=True).item()

    if key not in data:
        raise KeyError(
            f"Key '{key}' not found in SeerDrive prediction dict "
            f"({len(data)} trajectories total)."
        )

    arr = np.asarray(data[key])
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise RuntimeError(
            f"Trajectory for key '{key}' has unexpected shape {arr.shape}, "
            "expected (N,3) = [t, x, y]."
        )

    t = arr[:, 0]
    x_local = arr[:, 1]
    y_local = arr[:, 2]

    print(f"  SeerDrive traj for key '{key}': shape {arr.shape}")
    print("  First few samples [t, x, y]:")
    for i in range(min(5, len(arr))):
        print(f"    {t[i]:.3f}, {x_local[i]:.3f}, {y_local[i]:.3f}")

    return t, x_local, y_local


# -------------------------------------------------------------------------
# Centerline trimming
# -------------------------------------------------------------------------

def trim_centerline_to_gt(cl_x, cl_y, tr_x, tr_y):
    """
    Shorten the centerline so it spans only between the points closest to
    the start and end of the GT trajectory.

    This keeps the geometry identical, just slices the segment:
        cl_x[i0:i1+1], cl_y[i0:i1+1]
    """
    cl_x = np.asarray(cl_x)
    cl_y = np.asarray(cl_y)
    tr_x = np.asarray(tr_x)
    tr_y = np.asarray(tr_y)

    if len(cl_x) == 0 or len(tr_x) == 0:
        return cl_x, cl_y  # nothing to do

    cl_pts = np.stack([cl_x, cl_y], axis=1)

    start = np.array([tr_x[0], tr_y[0]])
    end = np.array([tr_x[-1], tr_y[-1]])

    d0 = np.linalg.norm(cl_pts - start, axis=1)
    d1 = np.linalg.norm(cl_pts - end, axis=1)

    i0 = int(np.argmin(d0))
    i1 = int(np.argmin(d1))

    if i0 > i1:
        i0, i1 = i1, i0

    # In degenerate cases where indices are equal, at least keep a small slice
    if i0 == i1:
        i1 = min(i0 + 1, len(cl_x) - 1)

    cl_x_short = cl_x[i0:i1 + 1]
    cl_y_short = cl_y[i0:i1 + 1]

    print(f"\n🔹 Trimming centerline to GT span:")
    print(f"  Original centerline length: {len(cl_x)}")
    print(f"  Trimmed centerline indices: [{i0}:{i1}] → length {len(cl_x_short)}")

    return cl_x_short, cl_y_short


# -------------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------------

def plot_map_centerline_traj(
    polygons,
    centerline_xy,
    traj_xy,
    out_png: str,
    title: str = "",
    pred_xy=None,
):
    """
    Plot drivable area map + centerline + GT trajectory (+ optional prediction)
    and save to out_png.
    """
    cl_x, cl_y = centerline_xy
    tr_x, tr_y = traj_xy

    plt.figure(figsize=(8, 8))

    # Drivable polygons
    for poly in polygons:
        x, y = poly.exterior.xy
        plt.fill(x, y, alpha=0.3, facecolor="lightgray", edgecolor="gray", label=None)

    # Centerline
    plt.plot(cl_x, cl_y, linestyle="-", linewidth=2.0, color="blue", label="Centerline")

    # GT trajectory
    plt.plot(
        tr_x,
        tr_y,
        linestyle="-",
        marker="o",
        markersize=4,
        color="red",
        label="GT trajectory",
    )

    # Predicted trajectory (if provided)
    if pred_xy is not None:
        px, py = pred_xy
        plt.plot(
            px,
            py,
            linestyle="-",
            marker="o",
            markersize=4,
            color="green",
            label="SeerDrive prediction",
        )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title if title else "Drivable map + centerline + trajectories")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)

    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"\n✅ Saved plot: {out_png}\n")

    plt.show()


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python print_gt_from_metric_cache.py <metric_cache_file>")
        print("  python print_gt_from_metric_cache.py <metric_cache_file> <seerdrive_key>")
        sys.exit(1)

    mc_path = sys.argv[1]
    seer_key = sys.argv[2] if len(sys.argv) >= 3 else None

    print(f"\n📦 Loading MetricCache from:\n{mc_path}\n")

    mc = load_metric_cache(mc_path)
    print(f"Loaded object type: {type(mc)}")

    # Extract components
    print("\n🔹 Extracting drivable_area_map polygons...")
    polygons = extract_drivable_polygons(mc)
    print(f"  Found {len(polygons)} polygons.")

    print("\n🔹 Extracting centerline...")
    cl_x, cl_y = extract_centerline_xy(mc)
    print(f"  Centerline length (full): {len(cl_x)} points.")

    print("\n🔹 Sampling GT trajectory...")
    tr_x, tr_y = sample_trajectory_xy(mc)
    print(f"  GT trajectory length: {len(tr_x)} samples.")

    # --- trim centerline to GT span ---
    cl_x_short, cl_y_short = trim_centerline_to_gt(cl_x, cl_y, tr_x, tr_y)

    pred_xy = None
    if seer_key is not None:
        print(f"\n🔹 Loading SeerDrive prediction for key: {seer_key}")
        t, x_loc, y_loc = load_seerdrive_trajectory(seer_key)

        print("\n🔹 Getting ego pose for local→global transform...")
        x0, y0, yaw0 = get_ego_pose_xyh(mc)
        print(f"  Ego pose at start: x0={x0:.3f}, y0={y0:.3f}, yaw0={yaw0:.3f} rad")

        x_pred_g, y_pred_g = local_to_global(x_loc, y_loc, x0, y0, yaw0)
        print("  First few global prediction points:")
        for i in range(min(5, len(x_pred_g))):
            print(f"    X={x_pred_g[i]:.3f}, Y={y_pred_g[i]:.3f}")

        pred_xy = (x_pred_g, y_pred_g)

    # Build output filename
    base = os.path.splitext(os.path.basename(mc_path))[0]
    if seer_key is not None:
        out_png = f"{base}_map_centerline_gt_pred_{seer_key}.png"
    else:
        out_png = f"{base}_map_centerline_gt.png"

    title = f"MetricCache: {base}"
    plot_map_centerline_traj(
        polygons,
        (cl_x_short, cl_y_short),  # use trimmed centerline
        (tr_x, tr_y),
        out_png,
        title=title,
        pred_xy=pred_xy,
    )

    print("🎉 Done.\n")


if __name__ == "__main__":
    main()
