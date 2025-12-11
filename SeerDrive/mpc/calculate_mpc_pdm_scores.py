"""
Script to calculate PDM scores for all MPC trajectories stored in mpc_trajectory directory.
Saves results to a CSV file with trajectory_key and all PDM score components.

Usage:
    python calculate_mpc_pdm_scores.py metric_cache_path=/path/to/metric_cache

    Or set the NAVSIM_EXP_ROOT environment variable and use:
    python calculate_mpc_pdm_scores.py

Requirements:
    - MPC trajectories should be stored in mpc_trajectory.npy (single file with dictionary)
    - The file should contain a dictionary where keys are trajectory_keys and values are numpy arrays
    - Each trajectory array should be of shape (N, 3) with [x, y, heading]
    - Metric cache must exist for each trajectory_key (token)
    
Output:
    - CSV file: mpc_pdm_scores_{timestamp}.csv with columns:
      * trajectory_key: The trajectory identifier
      * valid: Whether scoring was successful
      * score: Overall PDM score
      * no_at_fault_collisions: Collision metric
      * drivable_area_compliance: Drivable area metric
      * driving_direction_compliance: Driving direction metric
      * ego_progress: Progress metric
      * time_to_collision_within_bound: TTC metric
      * comfort: Comfort metric
"""

import os
import numpy as np
import pandas as pd
import lzma
import pickle
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import asdict
from datetime import datetime
import traceback
import logging

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from navsim.common.dataloader import MetricCacheLoader
from navsim.common.dataclasses import Trajectory, TrajectorySampling
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG_PATH = "navsim/planning/script/config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"


# Removed load_mpc_trajectory function - trajectories are now loaded directly from dictionary


def numpy_to_trajectory(poses: np.ndarray, interval_length: float = 1.0) -> Trajectory:
    """
    Convert numpy array of poses to Trajectory object.
    
    Args:
        poses: numpy array of shape (N, 3) with columns [x, y, heading]
        interval_length: time interval between poses in seconds (default 1.0 for 8 seconds / 8 poses)
    
    Returns:
        Trajectory object
    """
    # Ensure poses are float32
    poses = poses.astype(np.float32)
    
    # Create TrajectorySampling
    num_poses = poses.shape[0]
    trajectory_sampling = TrajectorySampling(
        num_poses=num_poses,
        interval_length=interval_length,
        time_horizon=interval_length * (num_poses - 1)
    )
    
    return Trajectory(poses=poses, trajectory_sampling=trajectory_sampling)


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main function to calculate PDM scores for all MPC trajectories.
    """
    # Get MPC trajectory file (single .npy file with dictionary)
    mpc_trajectory_file = Path(os.getcwd()) / "mpc_trajectory.npy"
    
    if not mpc_trajectory_file.exists():
        logger.error(f"MPC trajectory file not found: {mpc_trajectory_file}")
        logger.info("Expected file: mpc_trajectory.npy in current working directory")
        return
    
    # Load all MPC trajectories from the single file
    logger.info(f"Loading MPC trajectories from: {mpc_trajectory_file}")
    mpc_trajectories_dict = np.load(mpc_trajectory_file, allow_pickle=True).item()
    logger.info(f"Loaded {len(mpc_trajectories_dict)} trajectories from file")
    
    # Initialize simulator and scorer
    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    assert simulator.proposal_sampling == scorer.proposal_sampling, \
        "Simulator and scorer proposal sampling must be identical"
    
    # Load metric cache loader
    metric_cache_path = Path(cfg.metric_cache_path)
    if not metric_cache_path.exists():
        logger.error(f"Metric cache path not found: {metric_cache_path}")
        logger.info("Please set the metric_cache_path in the config or as an override")
        return
    
    metric_cache_loader = MetricCacheLoader(metric_cache_path)
    logger.info(f"Loaded metric cache with {len(metric_cache_loader)} tokens")
    
    # Calculate PDM scores for each trajectory
    results: List[Dict[str, Any]] = []
    
    for trajectory_key, mpc_poses in mpc_trajectories_dict.items():
        logger.info(f"Processing trajectory: {trajectory_key}")
        
        score_row: Dict[str, Any] = {
            "trajectory_key": trajectory_key,
            "valid": False
        }
        
        try:
            # mpc_poses is already loaded from the dictionary
            logger.info(f"  Loaded trajectory with shape: {mpc_poses.shape}")
            
            # Check if metric cache exists for this token
            if trajectory_key not in metric_cache_loader.tokens:
                logger.warning(f"  No metric cache found for token: {trajectory_key}")
                score_row["error"] = "No metric cache found"
                results.append(score_row)
                continue
            
            # Load metric cache
            metric_cache_path_token = metric_cache_loader.metric_cache_paths[trajectory_key]
            with lzma.open(metric_cache_path_token, "rb") as f:
                metric_cache = pickle.load(f)
            
            # Convert numpy array to Trajectory object
            # Assuming 8 seconds duration with 8 poses = 1.0 second interval
            # Adjust interval_length based on your actual trajectory duration
            num_poses = mpc_poses.shape[0]
            interval_length = 8.0 / num_poses if num_poses > 0 else 1.0  # 8 seconds total
            
            model_trajectory = numpy_to_trajectory(mpc_poses, interval_length=interval_length)
            
            # Calculate PDM score
            pdm_result = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=model_trajectory,
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
            )
            
            # Add scores to result row
            score_row.update(asdict(pdm_result))
            score_row["valid"] = True
            
            logger.info(f"  PDM Score: {pdm_result.score:.4f}")
            
        except Exception as e:
            logger.error(f"  Error processing {trajectory_key}: {str(e)}")
            logger.error(traceback.format_exc())
            score_row["error"] = str(e)
        
        results.append(score_row)
    
    # Create DataFrame and save to CSV
    if results:
        df = pd.DataFrame(results)
        
        # Reorder columns to put trajectory_key first
        cols = ["trajectory_key", "valid"] + [c for c in df.columns if c not in ["trajectory_key", "valid", "error"]]
        if "error" in df.columns:
            cols.append("error")
        df = df[cols]
        
        # Save to CSV
        timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
        output_path = Path(os.getcwd()) / f"mpc_pdm_scores_{timestamp}.csv"
        df.to_csv(output_path, index=False)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Results saved to: {output_path}")
        logger.info(f"{'='*60}")
        
        # Print summary
        valid_results = df[df["valid"] == True]
        if len(valid_results) > 0:
            logger.info(f"\nSummary:")
            logger.info(f"  Total trajectories: {len(df)}")
            logger.info(f"  Successfully scored: {len(valid_results)}")
            logger.info(f"  Failed: {len(df) - len(valid_results)}")
            
            if "score" in valid_results.columns:
                avg_score = valid_results["score"].mean()
                logger.info(f"  Average PDM Score: {avg_score:.4f}")
        else:
            logger.warning("No valid results to summarize")
    else:
        logger.warning("No results to save")


if __name__ == "__main__":
    main()

