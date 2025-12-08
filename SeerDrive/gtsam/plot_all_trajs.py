#!/usr/bin/env python3
"""
plot_all_trajs.py

Usage:
    python plot_all_trajs.py path/to/metric_cache.pkl <seerdrive_key>

Description:
    1. Loads MetricCache (Map, Centerline, GT).
    2. Loads SeerDrive trajectory (Local) from fixed .npy.
    3. Optimizes SeerDrive trajectory using GTSAM (smoothing).
    4. Transforms everything to Global coordinates.
    5. Plots 4 trajectories: Centerline (clipped), GT, SeerDrive, GTSAM Optimized.
"""

import sys
import os
import lzma
import pickle
import numpy as np
import matplotlib.pyplot as plt

# GTSAM imports
import gtsam
from gtsam import symbol

from shapely.geometry import Polygon

from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.evaluate.pdm_score import get_trajectory_as_array
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

# -------------------------------------------------------------------------
# Fixed path to SeerDrive .npy
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
    """Extract shapely Polygons from metric_cache.drivable_area_map."""
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
    """Extract centerline coordinates from metric_cache.centerline."""
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


def sample_gt_trajectory_xy(metric_cache: MetricCache):
    """Sample the ground-truth ego trajectory from metric_cache.trajectory."""
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
    """Get ego pose (x, y, yaw) at the start from the MetricCache's ego_state."""
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


def local_to_global_xy(x_local, y_local, x0, y0, yaw0):
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


def load_seerdrive_trajectory_xy(key: str):
    """
    Load SeerDrive predicted trajectory for a given key/token from the fixed .npy file.
    Assumes each trajectory row is [x_local, y_local, heading].
    """
    print(f"\n🔹 Loading SeerDrive trajectories from:\n{SEERDRIVE_TRAJ_NPY}")
    data = np.load(SEERDRIVE_TRAJ_NPY, allow_pickle=True).item()

    if key not in data:
        raise KeyError(f"Key '{key}' not found in SeerDrive prediction dict.")

    arr = np.asarray(data[key])
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise RuntimeError(f"Trajectory for key '{key}' has unexpected shape {arr.shape}.")

    x_local = arr[:, 0]
    y_local = arr[:, 1]
    
    return x_local, y_local

# -------------------------------------------------------------------------
# GTSAM Optimization Logic
# -------------------------------------------------------------------------

def build_noise(sigmas):
    sigmas = np.array(sigmas, dtype=float)
    return gtsam.noiseModel.Diagonal.Sigmas(sigmas)

def optimize_trajectory(
    seer_traj: np.ndarray,
    sigma_prior: float = 0.05,
    sigma_seer: float = 0.5,
    sigma_vel: float = 1.0,
    max_iters: int = 50,
) -> np.ndarray:
    """
    Optimizes a 2D trajectory (N, 2) using GTSAM factor graph.
    """
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

    # Priors for each point to stay close to original prediction
    for k, (x, y) in enumerate(seer_traj):
        graph.add(gtsam.PriorFactorPoint2(
            symbol('x', k),
            gtsam.Point2(float(x), float(y)),
            seer_noise
        ))

    # Smoothness (BetweenFactors)
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
# Centerline trimming
# -------------------------------------------------------------------------

def path_length(xs, ys):
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    if len(xs) < 2: return 0.0
    return float(np.sum(np.hypot(np.diff(xs), np.diff(ys))))

def trim_centerline_to_gt_length(cl_x, cl_y, tr_x, tr_y):
    """Shorten centerline to start near GT start and match GT length."""
    cl_x = np.asarray(cl_x)
    cl_y = np.asarray(cl_y)
    tr_x = np.asarray(tr_x)
    tr_y = np.asarray(tr_y)

    if len(cl_x) == 0 or len(tr_x) == 0:
        return cl_x, cl_y 

    cl_pts = np.stack([cl_x, cl_y], axis=1)
    start = np.array([tr_x[0], tr_y[0]])

    # index of centerline point closest to GT start
    d0 = np.linalg.norm(cl_pts - start, axis=1)
    i0 = int(np.argmin(d0))

    gt_len = path_length(tr_x, tr_y)

    seg_x = cl_x[i0:]
    seg_y = cl_y[i0:]
    if len(seg_x) < 2: return cl_x, cl_y

    seg_d = np.hypot(np.diff(seg_x), np.diff(seg_y))
    cum_d = np.concatenate([[0.0], np.cumsum(seg_d)])

    j_rel = int(np.searchsorted(cum_d, gt_len, side="right"))
    j_rel = min(j_rel, len(seg_x) - 1)
    i1 = i0 + j_rel

    print("\n🔹 Trimming centerline to GT spatial length:")
    print(f"  Trim indices: [{i0}:{i1}]")

    return cl_x[i0:i1 + 1], cl_y[i0:i1 + 1]

# -------------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------------

def plot_all(polygons, centerline_xy, gt_xy, pred_xy, opt_xy, out_png, title=""):
    """
    Plot Drivable Map, Centerline, GT, SeerDrive (Raw), and GTSAM (Opt).
    """
    cl_x, cl_y = centerline_xy
    gt_x, gt_y = gt_xy
    pr_x, pr_y = pred_xy
    opt_x, opt_y = opt_xy

    fig, ax = plt.subplots(figsize=(10, 10))

    # 1. Drivable polygons
    for poly in polygons:
        x_poly, y_poly = poly.exterior.xy
        ax.fill(x_poly, y_poly, alpha=0.3, facecolor="lightgray", edgecolor="gray", linewidth=0.5)

    # 2. Centerline
    ax.plot(cl_x, cl_y, linestyle="-", linewidth=2.0, color='black', alpha=0.6, label="Centerline")

    # 3. GT trajectory
    ax.plot(gt_x, gt_y, linestyle="-", marker="o", markersize=5, color='green', label="Ground Truth")

    # 4. SeerDrive prediction (Raw)
    ax.plot(pr_x, pr_y, linestyle="--", marker=".", markersize=4, color='blue', alpha=0.7, label="SeerDrive (Raw)")

    # 5. GTSAM Optimized
    ax.plot(opt_x, opt_y, linestyle="-", marker="x", markersize=5, color='red', linewidth=1.5, label="SeerOpt GTSAM")

    # Axis limits
    all_x = np.concatenate([cl_x, gt_x, pr_x, opt_x])
    all_y = np.concatenate([cl_y, gt_y, pr_y, opt_y])

    x_min, x_max = np.min(all_x), np.max(all_x)
    y_min, y_max = np.min(all_y), np.max(all_y)

    span = max(x_max - x_min, y_max - y_min)
    margin = 0.2 * span
    x_mid, y_mid = (x_min + x_max) / 2, (y_min + y_max) / 2

    ax.set_xlim(x_mid - span/2 - margin, x_mid + span/2 + margin)
    ax.set_ylim(y_mid - span/2 - margin, y_mid + span/2 + margin)
    ax.set_aspect('equal', adjustable='box')

    ax.set_xlabel("Global X [m]")
    ax.set_ylabel("Global Y [m]")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"\n✅ Saved plot: {out_png}\n")
    plt.show()

# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    if len(sys.argv) != 3:
        print("Usage: python plot_all_trajs.py <metric_cache_file> <seerdrive_key>")
        sys.exit(1)

    mc_path = sys.argv[1]
    seer_key = sys.argv[2]

    # 1. Load MetricCache
    print(f"\n📦 Loading MetricCache from:\n{mc_path}")
    mc = load_metric_cache(mc_path)

    # 2. Extract Data
    print("\n🔹 Extracting Map and Trajectories...")
    polygons = extract_drivable_polygons(mc)
    gt_x, gt_y = sample_gt_trajectory_xy(mc)
    cl_x_full, cl_y_full = extract_centerline_xy(mc)
    cl_x, cl_y = trim_centerline_to_gt_length(cl_x_full, cl_y_full, gt_x, gt_y)

    # 3. Load SeerDrive (Local Frame)
    print(f"\n🔹 Loading SeerDrive trajectory for key: {seer_key}")
    sx_local, sy_local = load_seerdrive_trajectory_xy(seer_key)

    # 4. Run GTSAM Optimization (in Local Frame)
    print("\n🔹 Running GTSAM optimization on SeerDrive trajectory...")
    seer_local_arr = np.column_stack((sx_local, sy_local)) # Stack to (N,2)
    
    # You can tune sigmas here if needed
    opt_local_arr = optimize_trajectory(
        seer_local_arr, 
        sigma_prior=0.05, 
        sigma_seer=0.5, 
        sigma_vel=1.0
    )
    ox_local, oy_local = opt_local_arr[:, 0], opt_local_arr[:, 1]
    print(f"  Optimized {len(ox_local)} points.")

    # 5. Transform to Global Frame
    print("\n🔹 transforming trajectories to Global Frame...")
    x0, y0, yaw0 = get_ego_pose_xyh(mc)
    print(f"  Ego Start: x={x0:.2f}, y={y0:.2f}, yaw={yaw0:.2f}")

    # Transform Raw SeerDrive
    sx_global, sy_global = local_to_global_xy(sx_local, sy_local, x0, y0, yaw0)
    # Transform GTSAM Optimized
    ox_global, oy_global = local_to_global_xy(ox_local, oy_local, x0, y0, yaw0)

    # Truncate to match GT length for cleaner plotting
    n = min(len(gt_x), len(sx_global))
    sx_global, sy_global = sx_global[:n], sy_global[:n]
    ox_global, oy_global = ox_global[:n], oy_global[:n]

    # 6. Plotting
    base = os.path.splitext(os.path.basename(mc_path))[0]
    out_png = f"{base}_all_trajs_{seer_key}.png"
    title = f"Map + Centerline + GT + SeerDrive + GTSAM\nKey: {seer_key}"

    plot_all(
        polygons, 
        (cl_x, cl_y), 
        (gt_x, gt_y), 
        (sx_global, sy_global), 
        (ox_global, oy_global),
        out_png, 
        title=title
    )

if __name__ == "__main__":
    main()