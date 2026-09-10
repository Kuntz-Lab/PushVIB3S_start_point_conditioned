"""One-off / occasionally-rerun generator for test_deploy_gold_standard.py's fixtures.

Picks a few real episodes, freezes their point clouds as fixtures under unit_test_data/, then
runs DeployPushVIB3S.predict_push on them with a specific checkpoint to produce the "gold
standard" predictions that test_deploy_gold_standard.py checks against on every run.

Only re-run this script when you WANT to move the gold standard (e.g. you intentionally
retrained/changed the architecture and have manually verified the new predictions are
correct). Running it overwrites unit_test_data/gold_standard_predictions.json.
"""

from pathlib import Path
import json
import pickle

import numpy as np

from deploy import DeployPushVIB3S
from PushVIB3S_start_point_conditioned import ENCODER_BPS
from utils import find_pickle_files

# --- edit these to change what the gold standard locks in ---
WEIGHTS_PATH = (
    "checkpoints/realsense_bg0.0005_weight1.5_w_no_diff_dataend_point_bps_local_"
    "20260827_131126/checkpoint_429_0.020384.pt"
)
BASIS_PATH = "bps/bps_basis_r1.0_n128.npy"
ENCODER_TYPE = ENCODER_BPS
SOURCE_TEST_DIR = "/home/britton/PushVIBES/data/all_good_realsense_data/test"
EPISODE_INDICES = [0, 33, 66]  # which episodes (sorted find_pickle_files order) to freeze
OUTPUT_DIR = Path("unit_test_data")
GOLD_STANDARD_JSON_PATH = OUTPUT_DIR / "gold_standard_predictions.json"
SEED_BASE = 20260827  # per-episode seed = SEED_BASE + episode index in EPISODE_INDICES
DEVICE = "cpu"  # fixed (not cuda) so fixtures/gold standard are reproducible on any machine
# ----------------------------------------------------------------


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    episode_paths = find_pickle_files(SOURCE_TEST_DIR)
    if not episode_paths:
        raise FileNotFoundError(f"No pickle files found in {SOURCE_TEST_DIR}")

    deployer = DeployPushVIB3S(
        weights_path=WEIGHTS_PATH,
        basis_path=BASIS_PATH,
        device=DEVICE,
        encoder_type=ENCODER_TYPE,
    )

    gold_standard = {
        "weights_path": WEIGHTS_PATH,
        "basis_path": BASIS_PATH,
        "encoder_type": ENCODER_TYPE,
        "seed_base": SEED_BASE,
        "device": DEVICE,
        "episodes": [],
    }

    for fixture_idx, episode_idx in enumerate(EPISODE_INDICES):
        episode_path = episode_paths[episode_idx]
        with open(episode_path, "rb") as f:
            episode = pickle.load(f)

        start_pointcloud = np.asarray(episode["start_pointcloud"], dtype=np.float32)
        goal_pointcloud = np.asarray(episode["goal_pointcloud"], dtype=np.float32)
        push_start_point = np.asarray(episode["start_position"], dtype=np.float32).reshape(3)

        fixture_path = OUTPUT_DIR / f"episode_{fixture_idx}.npz"
        np.savez_compressed(
            fixture_path,
            start_pointcloud=start_pointcloud,
            goal_pointcloud=goal_pointcloud,
            push_start_point=push_start_point,
            source_path=str(episode_path),
        )

        seed = SEED_BASE + fixture_idx
        np.random.seed(seed)
        predicted_end_point, predicted_displacement = deployer.predict_push(
            start_pointcloud, goal_pointcloud, push_start_point
        )

        gold_standard["episodes"].append(
            {
                "fixture_file": fixture_path.name,
                "source_path": str(episode_path),
                "seed": seed,
                "predicted_end_point": predicted_end_point.tolist(),
                "predicted_displacement": predicted_displacement.tolist(),
            }
        )

        print(f"[{fixture_idx}] {episode_path}")
        print(f"    predicted_end_point:    {predicted_end_point}")
        print(f"    predicted_displacement: {predicted_displacement}")

    with open(GOLD_STANDARD_JSON_PATH, "w") as f:
        json.dump(gold_standard, f, indent=2)

    print(f"\nWrote {len(EPISODE_INDICES)} fixtures to {OUTPUT_DIR}/")
    print(f"Wrote gold standard predictions to {GOLD_STANDARD_JSON_PATH}")


if __name__ == "__main__":
    main()
