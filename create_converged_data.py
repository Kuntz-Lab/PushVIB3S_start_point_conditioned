"""
Generate "no diff goal" training data: synthetic samples where the object is
already at its goal, so the model learns to predict zero displacement.

For a random subset of the pickles found under a source data directory, each
new sample is built from an existing one as follows:
  - new start_pointcloud = new goal_pointcloud = original goal_pointcloud
  - new start_position   = original end point (start_position + displacement)
  - new displacement     = zeros (new start point == new end point)

Results are written to a "no_diff_goal_data" subdirectory created inside the
source data directory, mirroring the source's relative folder structure.

Works with either of the two known source layouts, since both expose the
same 'start_pointcloud' / 'goal_pointcloud' / 'start_position' / 'displacement'
keys consumed by dataset.py:
  - /home/britton/PushVIBES/data/aug7_2025_new_data/processed_data_noisy_object_frame_with_old_and_new_data
  - /home/britton/PushVIBES/data/all_good_realsense_data/training
"""

import os
import pickle
import random

import numpy as np

# Source data directory to process. Either of the two known layouts works:
DATA_DIR = "/home/britton/PushVIBES/data/aug7_2025_new_data/processed_data_noisy_object_frame_with_old_and_new_data"
# DATA_DIR = "/home/britton/PushVIBES/data/all_good_realsense_data/training"

# Fraction (0-1) of the source pickle files to convert into no-diff-goal data.
RATIO_OF_DATA_TO_PROCESS = 0.5

# Random seed used when sampling which files to process.
SEED = 14

OUTPUT_DIRNAME = "no_diff_goal_data"


def find_source_pickle_files(data_dir, output_dir):
    """Recursively find .pickle/.pkl files under data_dir, skipping output_dir."""
    output_dir_abs = os.path.abspath(output_dir)
    file_list = []
    for root, dirs, files in os.walk(data_dir):
        if os.path.abspath(root) == output_dir_abs:
            dirs[:] = []  # don't descend into the directory we're writing to
            continue
        for file in files:
            if file.endswith((".pickle", ".pkl")):
                file_list.append(os.path.join(root, file))
    return file_list


def make_no_diff_goal_sample(data):
    goal_pointcloud = data["goal_pointcloud"]
    start_position = data["start_position"]
    displacement = data["displacement"]

    end_position = start_position + displacement

    return {
        "start_pointcloud": np.copy(goal_pointcloud),
        "goal_pointcloud": np.copy(goal_pointcloud),
        "start_position": end_position,
        "displacement": np.zeros_like(displacement),
    }


def process_directory(data_dir, ratio_of_data_to_process=RATIO_OF_DATA_TO_PROCESS, seed=0):
    output_dir = os.path.join(data_dir, OUTPUT_DIRNAME)
    os.makedirs(output_dir, exist_ok=True)

    source_files = sorted(find_source_pickle_files(data_dir, output_dir))
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

    for src_path in selected_files:
        with open(src_path, "rb") as f:
            data = pickle.load(f)

        new_data = make_no_diff_goal_sample(data)

        rel_path = os.path.relpath(src_path, data_dir)
        rel_dir, filename = os.path.split(rel_path)
        name, ext = os.path.splitext(filename)

        out_subdir = os.path.join(output_dir, rel_dir)
        os.makedirs(out_subdir, exist_ok=True)
        out_path = os.path.join(out_subdir, f"{name}_no_diff{ext}")

        with open(out_path, "wb") as f:
            pickle.dump(new_data, f)

    print(f"Wrote {len(selected_files)} no-diff-goal pickles to {output_dir}")


if __name__ == "__main__":
    process_directory(DATA_DIR, ratio_of_data_to_process=RATIO_OF_DATA_TO_PROCESS, seed=SEED)
