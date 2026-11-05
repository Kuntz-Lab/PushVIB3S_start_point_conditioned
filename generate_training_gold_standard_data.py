"""Writes the gold standard that test_training_gold_standard.py checks against.

Run from the repo root (needs CUDA):
    python generate_training_gold_standard_data.py

initial_weights.pt is only written if missing (delete it on purpose to re-derive). Expected
weights + histories are rewritten for every case in CASES_TO_REGENERATE. Only re-run when you
WANT to move the gold standard; review `git diff --stat unit_test_data/` before committing.
"""

import torch

from PushVIB3S_start_point_conditioned import PushVIB3S_start_point_conditioned
from training_gold_standard_common import (
    TRAINER_KWARGS, TRAINING_CASES, TRAINING_INITIAL_WEIGHTS_PATH, TRAINING_INIT_SEED,
    check_environment, environment_info, run_training, save_json,
)
from utils import set_seed


if __name__ == "__main__":
    # --- control variables ---
    CASES_TO_REGENERATE = list(TRAINING_CASES)  # keys of TRAINING_CASES
    # --------------------------

    check_environment()

    if not TRAINING_INITIAL_WEIGHTS_PATH.exists():
        set_seed(TRAINING_INIT_SEED)
        model_kwargs = {k: TRAINER_KWARGS[k] for k in ["n_neurons", "in_bps", "goalD", "feature_dim",
                                                       "use_directional_bps", "encoder_type"]}
        torch.save(PushVIB3S_start_point_conditioned(**model_kwargs).state_dict(), TRAINING_INITIAL_WEIGHTS_PATH)

    for case in CASES_TO_REGENERATE:
        trainer_kwargs, weights_path, history_path = TRAINING_CASES[case]
        state_dict, history = run_training(trainer_kwargs)
        torch.save(state_dict, weights_path)
        save_json({"environment": environment_info(), "history": history}, history_path)
        print(f"[{case}] wrote {weights_path} and {history_path}: {history['checkpoint_files']}")
