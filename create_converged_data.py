"""
Generate "no diff goal" training data: synthetic samples where the object is
already at its goal, so the model learns to predict zero displacement.

For a random subset of the pickles found under DATA_DIR (original, non object
frame data), each new sample is built from an existing one as follows:
  - new start_pc     = new goal_pc   = original goal_pc
  - new start_pose   = new goal_pose = original goal_pose
  - new displacement = {x: 0, y: 0, z: 0}
  - any other keys (e.g. context_pc, exposure_area, exposure_overlay_image) are
    copied through unchanged

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

from utils import find_pickle_files

# Source data directory to process (original frame, not object frame).
DATA_DIR = "/home/britton/data/PushVIBES/PushVIBES_realworld_data/source_of_truth_original_frame/training"

# Fraction (0-1) of the source pickle files to convert into no-diff-goal data.
RATIO_OF_DATA_TO_PROCESS = 0.5

# Random seed used when sampling which files to process.
SEED = 14

OUTPUT_DIR = "/home/britton/data/PushVIBES/PushVIBES_realworld_data/original_frame_no_diff_subsets/seed14_ratio_0.5/training"

REQUIRED_KEYS = ("start_pc", "goal_pc", "start_pose", "goal_pose", "displacement")


def _is_within(path, parent):
    path = os.path.realpath(path)
    parent = os.path.realpath(parent)
    return os.path.commonpath([path, parent]) == parent


def make_no_diff_goal_sample(data):
    missing = [k for k in REQUIRED_KEYS if k not in data]
    if missing:
        raise KeyError(f"Sample is missing required keys: {missing}")

    # Deep copy so nothing in the new sample shares memory with the source.
    new_data = copy.deepcopy(data)
    new_data["start_pc"] = copy.deepcopy(data["goal_pc"])
    new_data["goal_pc"] = copy.deepcopy(data["goal_pc"])
    new_data["start_pose"] = copy.deepcopy(data["goal_pose"])
    new_data["goal_pose"] = copy.deepcopy(data["goal_pose"])
    new_data["displacement"] = {axis: 0.0 for axis in data["displacement"]}
    return new_data


def process_directory(data_dir, output_dir=OUTPUT_DIR, ratio_of_data_to_process=RATIO_OF_DATA_TO_PROCESS, seed=0):
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
        f"Processing {len(selected_files)} files "
        f"({ratio_of_data_to_process:.0%}) into {output_dir}"
    )

    os.makedirs(output_dir, exist_ok=True)

    for src_path in selected_files:
        with open(src_path, "rb") as f:
            data = pickle.load(f)

        new_data = make_no_diff_goal_sample(data)

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
    process_directory(DATA_DIR, OUTPUT_DIR, ratio_of_data_to_process=RATIO_OF_DATA_TO_PROCESS, seed=SEED)
