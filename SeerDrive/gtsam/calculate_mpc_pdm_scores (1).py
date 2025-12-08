#!/usr/bin/env python3
"""
calculate_pdm_score_v2.py

Usage:
    python calculate_pdm_score_v2.py metric_cache_path=/path/to/metric_cache_folder

Description:
    Calculates PDM (Planner Driving Metric) scores for trajectories stored in a .npy file.
    Automatically detects navsim installation to find config files.
"""

import os
# Set default env var if not present
if "NAVSIM_EXP_ROOT" not in os.environ:
    os.environ["NAVSIM_EXP_ROOT"] = os.getcwd()

import sys
# ... rest of your imports
import numpy as np
import pandas as pd
import lzma
import pickle
import logging
import traceback
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import asdict
from datetime import datetime

# NavSim Imports
import navsim
from navsim.common.dataloader import MetricCacheLoader
from navsim.common.dataclasses import Trajectory, TrajectorySampling
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

# -------------------------------------------------------------------------
# Dynamic Configuration Path Finding
# -------------------------------------------------------------------------
# This block finds where 'navsim' is installed and sets the config path to that absolute location
# to prevent "Primary config directory not found" errors.

NAVSIM_ROOT = os.path.dirname(navsim.__file__)
# Path: <site-packages>/navsim/planning/script/config/pdm_scoring
POSSIBLE_CONFIG_PATH = "/scratch/rob535f25s001_class_root/rob535f25s001_class/kushkp/SeerOpt/SeerDrive/navsim/planning/script/config/pdm_scoring"

if os.path.exists(POSSIBLE_CONFIG_PATH):
    CONFIG_PATH = POSSIBLE_CONFIG_PATH
else:
    # Fallback: If you are running from the repo root, try relative
    CONFIG_PATH = "navsim/planning/script/config/pdm_scoring"

print(f"🔹 Using Hydra Config Path: {CONFIG_PATH}")

CONFIG_NAME = "default_run_pdm_score"

# -------------------------------------------------------------------------
# User Settings
# -------------------------------------------------------------------------

# FIXED PATH to your GTSAM input file
INPUT_TRAJ_FILE = "/scratch/rob535f25s001_class_root/rob535f25s001_class/kushkp/SeerOpt/SeerDrive/gtsam/seerdrive_traj_gtsamPreds.npy"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------

def numpy_to_trajectory(poses: np.ndarray, duration: float = 4.0) -> Trajectory:
    poses = poses.astype(np.float32)
    num_poses = poses.shape[0]

    if num_poses > 1:
        interval_length = duration / (num_poses - 1)
    else:
        interval_length = 0.5 

    trajectory_sampling = TrajectorySampling(
        num_poses=num_poses,
        interval_length=interval_length,
        time_horizon=duration
    )
    return Trajectory(poses=poses, trajectory_sampling=trajectory_sampling)

def estimate_headings_if_missing(traj: np.ndarray) -> np.ndarray:
    if traj.shape[1] == 3:
        return traj
    if traj.shape[1] != 2:
        raise ValueError(f"Trajectory shape {traj.shape} invalid. Expected (N,2) or (N,3).")

    x, y = traj[:, 0], traj[:, 1]
    dx = np.gradient(x)
    dy = np.gradient(y)
    headings = np.arctan2(dy, dx)
    return np.column_stack([x, y, headings])

# -------------------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------------------

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    
    # 1. Validation
    traj_path = Path(INPUT_TRAJ_FILE)
    if not traj_path.exists():
        logger.error(f"❌ Input file not found: {traj_path}")
        return

    # Handle Metric Cache Path
    metric_cache_path = Path(cfg.metric_cache_path)
    
    # If user pointed to a FILE, assume the parent folder is the cache directory
    if metric_cache_path.is_file():
        logger.warning(f"⚠️ You provided a file path for metric_cache: {metric_cache_path.name}")
        logger.warning(f"   Switching to parent directory: {metric_cache_path.parent}")
        metric_cache_path = metric_cache_path.parent

    if not metric_cache_path.exists():
        logger.error(f"❌ Metric cache path invalid: {metric_cache_path}")
        return

    # 2. Load Trajectories
    logger.info(f"📦 Loading GTSAM Trajectories from: {traj_path.name}")
    try:
        data = np.load(traj_path, allow_pickle=True)
        if isinstance(data, np.ndarray) and data.shape == () and data.dtype == object:
            trajectories_dict = data.item()
        elif isinstance(data, dict):
            trajectories_dict = data
        else:
            trajectories_dict = data.item() 
    except Exception as e:
        logger.error(f"❌ Failed to load .npy file: {e}")
        return

    logger.info(f"   Found {len(trajectories_dict)} trajectories.")

    # 3. Setup NavSim Components
    logger.info("🔧 Initializing PDM Simulator & Scorer...")
    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    metric_cache_loader = MetricCacheLoader(metric_cache_path)
    
    # 4. Scoring Loop
    results: List[Dict[str, Any]] = []

    for token, traj_array in trajectories_dict.items():
        row = {"token": token, "valid": False}
        
        if token not in metric_cache_loader.tokens:
            row["error"] = "Token missing in MetricCache"
            results.append(row)
            continue

        try:
            mc_path = metric_cache_loader.metric_cache_paths[token]
            with lzma.open(mc_path, "rb") as f:
                metric_cache = pickle.load(f)

            traj_array = np.array(traj_array)
            traj_array = estimate_headings_if_missing(traj_array)
            duration = getattr(metric_cache.trajectory, "duration", 4.0)
            model_traj = numpy_to_trajectory(traj_array, duration=duration)

            pdm_result = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=model_traj,
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
            )

            row.update(asdict(pdm_result))
            row["valid"] = True
            logger.info(f"   Token: {token[:8]}... | PDM Score: {pdm_result.score:.4f}")

        except Exception as e:
            logger.error(f"   Error on {token[:8]}: {e}")
            row["error"] = str(e)
        
        results.append(row)

    # 5. Output Results
    if not results:
        logger.warning("No results to save.")
        return

    df = pd.DataFrame(results)
    first_cols = ["token", "valid", "score"]
    cols = first_cols + [c for c in df.columns if c not in first_cols]
    df = df[[c for c in cols if c in df.columns]]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = f"pdm_scores_gtsam_{timestamp}.csv"
    df.to_csv(out_csv, index=False)

    logger.info("\n" + "="*50)
    logger.info(f"💾 RESULTS SAVED: {out_csv}")
    logger.info("="*50)

    valid_df = df[df["valid"] == True]
    if len(valid_df) > 0:
        logger.info(f"⭐ AVERAGE PDM SCORE: {valid_df['score'].mean():.5f}")
    else:
        logger.warning("No valid scores to average.")

if __name__ == "__main__":
    main()