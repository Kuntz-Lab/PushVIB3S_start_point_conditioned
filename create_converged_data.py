"""
Generate "no diff goal" training data: synthetic samples where the object is
already at its goal, so the model learns to predict zero displacement.

For a random subset of the pickles found under DATA_DIR (original, non object
frame data), each new sample is built from an existing one. DATA_TYPE selects
the source format:

  "real" (keys: start_pc, goal_pc, start_pose, goal_pose, displacement; poses
  and displacement are {x, y, z} dicts):
    - new start_pc     = new goal_pc   = original goal_pc
    - new start_pose   = new goal_pose = original goal_pose
    - new displacement = {x: 0, y: 0, z: 0}

  "sim" (keys: start_pointcloud, goal_pointcloud, global_goal_pointcloud,
  start_position, displacement; positions are (1, 3) arrays and the goal
  position is start_position + displacement):
    - new start_pointcloud = new goal_pointcloud = original goal_pointcloud
    - new start_position   = original goal position (start_position + displacement)
    - new displacement     = zeros

In both cases any other keys (e.g. context_pc, exposure_area,
exposure_overlay_image, global_goal_pointcloud) are copied through unchanged.

Results are written to OUTPUT_DIR, mirroring the source's relative folder
structure, with "_no_diff" appended to each filename.

The source data is never modified: source files are only opened read-only,
every value is deep-copied before use, OUTPUT_DIR may not overlap DATA_DIR,
and output files are opened in exclusive-create mode so nothing is overwritten.
"""

import copy
import os
import pickle
import random

import numpy as np

from utils import find_pickle_files

# Source data format: "real" or "sim".
DATA_TYPE = "sim"

# Source data directory to process (original frame, not object frame).
DATA_DIRS = {
    "real": "/home/britton/data/PushVIBES/PushVIBES_realworld_data/source_of_truth_original_frame/training",
    "sim": "/home/britton/data/PushVIBES/aug7_new_sim_data/data_fixed/original_frame",
}

# Fraction (0-1) of the source pickle files to convert into no-diff-goal data.
RATIO_OF_DATA_TO_PROCESS = 0.2

# Random seed used when sampling which files to process.
SEED = 14

OUTPUT_DIRS = {
    "real": "/home/britton/data/PushVIBES/PushVIBES_realworld_data/original_frame_no_diff_subsets/seed14_ratio_0.5/training",
    "sim": f"/home/britton/data/PushVIBES/aug7_new_sim_data/data_fixed/no_diff_subset_original_frame/seed{SEED}_ratio{RATIO_OF_DATA_TO_PROCESS}",
}

DATA_DIR = DATA_DIRS[DATA_TYPE]
OUTPUT_DIR = OUTPUT_DIRS[DATA_TYPE]

REQUIRED_KEYS = {
    "real": ("start_pc", "goal_pc", "start_pose", "goal_pose", "displacement"),
    "sim": ("start_pointcloud", "goal_pointcloud", "start_position", "displacement"),
}


def _is_within(path, parent):
    path = os.path.realpath(path)
    parent = os.path.realpath(parent)
    return os.path.commonpath([path, parent]) == parent


def _make_real_no_diff_goal_sample(data):
    # Deep copy so nothing in the new sample shares memory with the source.
    new_data = copy.deepcopy(data)
    new_data["start_pc"] = copy.deepcopy(data["goal_pc"])
    new_data["goal_pc"] = copy.deepcopy(data["goal_pc"])
    new_data["start_pose"] = copy.deepcopy(data["goal_pose"])
    new_data["goal_pose"] = copy.deepcopy(data["goal_pose"])
    new_data["displacement"] = {axis: 0.0 for axis in data["displacement"]}
    return new_data


def _make_sim_no_diff_goal_sample(data):
    # Deep copy so nothing in the new sample shares memory with the source.
    new_data = copy.deepcopy(data)
    new_data["start_pointcloud"] = np.copy(data["goal_pointcloud"])
    new_data["goal_pointcloud"] = np.copy(data["goal_pointcloud"])
    new_data["start_position"] = data["start_position"] + data["displacement"]
    new_data["displacement"] = np.zeros_like(data["displacement"])
    return new_data


def make_no_diff_goal_sample(data, data_type=DATA_TYPE):
    missing = [k for k in REQUIRED_KEYS[data_type] if k not in data]
    if missing:
        raise KeyError(
            f"Sample is missing required {data_type!r} keys: {missing} "
            f"(found {sorted(data)}). Is DATA_TYPE set correctly?"
        )

    if data_type == "real":
        return _make_real_no_diff_goal_sample(data)
    return _make_sim_no_diff_goal_sample(data)


def process_directory(data_dir, output_dir=OUTPUT_DIR, data_type=DATA_TYPE, ratio_of_data_to_process=RATIO_OF_DATA_TO_PROCESS, seed=0):
    if data_type not in REQUIRED_KEYS:
        raise ValueError(f"data_type must be one of {list(REQUIRED_KEYS)}, got {data_type!r}")
    if _is_within(output_dir, data_dir) or _is_within(data_dir, output_dir):
        raise ValueError(f"output_dir {output_dir} must not overlap data_dir {data_dir}")

    source_files = sorted(find_pickle_files(data_dir))
    if not source_files:
        print(f"No pickle files found under {data_dir}")
        return

    num_to_process = round(len(source_files) * ratio_of_data_to_process)
    num_to_process = max(0, min(num_to_process, len(source_files)))

    rng = random.Random(seed)
    selected_files = rng.sample(source_files, num_to_process)

    print(f"Found {len(source_files)} source pickle files under {data_dir}")
    print(
        f"Processing {len(selected_files)} {data_type} files "
        f"({ratio_of_data_to_process:.0%}) into {output_dir}"
    )

    os.makedirs(output_dir, exist_ok=True)

    for src_path in selected_files:
        with open(src_path, "rb") as f:
            data = pickle.load(f)

        new_data = make_no_diff_goal_sample(data, data_type)

        rel_path = os.path.relpath(src_path, data_dir)
        rel_dir, filename = os.path.split(rel_path)
        name, ext = os.path.splitext(filename)

        out_subdir = os.path.join(output_dir, rel_dir)
        out_path = os.path.join(out_subdir, f"{name}_no_diff{ext}")
        if _is_within(out_path, data_dir):
            raise ValueError(f"Refusing to write inside data_dir: {out_path}")

        os.makedirs(out_subdir, exist_ok=True)
        # "xb" fails if the file already exists, so nothing is ever overwritten.
        with open(out_path, "xb") as f:
            pickle.dump(new_data, f)

    print(f"Wrote {len(selected_files)} no-diff-goal pickles to {output_dir}")


if __name__ == "__main__":
    process_directory(DATA_DIR, OUTPUT_DIR, data_type=DATA_TYPE, ratio_of_data_to_process=RATIO_OF_DATA_TO_PROCESS, seed=SEED)
