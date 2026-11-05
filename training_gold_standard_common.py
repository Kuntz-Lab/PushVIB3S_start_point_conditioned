"""Shared config + code path for the training gold standard (backwards-compatibility) tests.

Used by both generate_training_gold_standard_data.py (writes the gold standard) and
test_training_gold_standard.py (checks against it), so both always run the exact same code.
Run from the repo root (checkpoints/ and bps/ are resolved relative to it, like train.py).
"""

import os

# cuBLAS needs this set before its first handle is created for deterministic GPU results
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from pathlib import Path
import json
import shutil

import numpy as np
import torch
import torch.multiprocessing as mp

from PushVIB3S_start_point_conditioned import STRATEGY_END_POINT, STRATEGY_DISPLACEMENT, ENCODER_BPS
from train import PushVibesTrainer
from utils import set_seed

torch.use_deterministic_algorithms(True)

REPO_ROOT = Path(__file__).resolve().parent
TRAINING_DIR = Path("unit_test_data") / "training"
# 512-point episodes (copied from PushVIBES/unit_test_data/training), so PushDataset skips its unseeded FPS
TRAINING_SIM_DIR = TRAINING_DIR / "sim"
TRAINING_REAL_TRAIN_DIR = TRAINING_DIR / "real_train"
TRAINING_REAL_VAL_DIR = TRAINING_DIR / "real_val"
TRAINING_INITIAL_WEIGHTS_PATH = TRAINING_DIR / "initial_weights.pt"  # fixed start weights
TRAINING_INIT_SEED = 342
TRAIN_SEED = 333  # set_seed() before building the trainer (like main()) and again right before trainer.train()

# Mirrors train.py main(), shrunk to a few tiny epochs
TRAINER_KWARGS = {
    "n_neurons": 512,
    "in_bps": 128,
    "goalD": 7,
    "feature_dim": 128,
    "beta_goal": 0.0005,
    "batch_size": 4,
    "learning_rate": 3e-4,
    "num_epochs": 5,  # > kl_warmup_epochs (3) + 1, so the post-warmup best-checkpoint path is exercised
    "seed": 342,
    "deterministic": False,
    "use_directional_bps": True,
    "sim_validation": False,
    "data_dirs": [str(TRAINING_SIM_DIR), str(TRAINING_REAL_TRAIN_DIR)],  # first source must be the sim data
    "data_source_weights": [1.0, 1.5],
    "data_pts_per_epoch": [8, -1],  # sim < len(sim) -> exercises the RandomSampler subsample path; all real data
    "validation_data_dir": str(TRAINING_REAL_VAL_DIR),
    "pretrained_weights_path": str(TRAINING_INITIAL_WEIGHTS_PATH),
    "run_name_prefix": "unit_test_gold_standard_",
    "encoder_type": ENCODER_BPS,
}

# One gold standard per case: (trainer kwargs, expected weights path, expected history path)
TRAINING_CASES = {
    "end_point": (dict(TRAINER_KWARGS, strategy=STRATEGY_END_POINT),
                  TRAINING_DIR / "expected_trained_weights_end_point.pt",
                  TRAINING_DIR / "expected_training_history_end_point.json"),
    "displacement": (dict(TRAINER_KWARGS, strategy=STRATEGY_DISPLACEMENT),
                     TRAINING_DIR / "expected_trained_weights_displacement.pt",
                     TRAINING_DIR / "expected_training_history_displacement.json"),
    # mixed_batches=True: 8 sim + 8 real per epoch shuffled into 4 batches of 4, real weighted 1.5 per sample
    "end_point_mixed_batches": (dict(TRAINER_KWARGS, strategy=STRATEGY_END_POINT, mixed_batches=True,
                                     run_name_prefix="unit_test_gold_standard_mixed_"),
                                TRAINING_DIR / "expected_trained_weights_end_point_mixed_batches.pt",
                                TRAINING_DIR / "expected_training_history_end_point_mixed_batches.json"),
}


def check_environment():
    """Fail early if not run from the repo root or without a GPU (train.py hardcodes cuda).
    Inputs: none. Output: none"""
    if Path.cwd().resolve() != REPO_ROOT:
        raise RuntimeError(f"Run the training gold-standard tests from the repo root ({REPO_ROOT}), not {Path.cwd()}")
    if not torch.cuda.is_available():
        raise RuntimeError("The training gold-standard tests need CUDA (train.py hardcodes device='cuda')")


def environment_info():
    """Versions/hardware the gold standard was generated on, for diagnosing mismatches.
    Inputs: none. Output: dict[str, str]"""
    return {"torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0), "numpy": np.__version__}


def run_training(trainer_kwargs):
    """Run a few epochs of PushVibesTrainer (spawned DataLoader workers, like train.py main()) and
    delete its checkpoint dir afterwards. Re-seeds right before train(), so a change that only
    alters how much RNG model construction consumes (e.g. adding/removing an unused head) doesn't
    move the result.
    Inputs: trainer_kwargs (dict). Output: (state_dict on CPU (dict[str, torch.Tensor]), history (dict))"""
    mp.set_start_method("spawn", force=True)
    set_seed(TRAIN_SEED)
    trainer = PushVibesTrainer(**trainer_kwargs)
    set_seed(TRAIN_SEED)
    trainer.train()

    state_dict = {k: v.detach().cpu().clone() for k, v in trainer.model.state_dict().items()}
    history = {
        "train_loss": trainer.train_loss_history,
        "real_val_loss": trainer.real_val_loss_history,
        "real_val_error": trainer.real_val_error_history,
        # Which epochs were saved as "best" (filenames encode epoch + loss)
        "checkpoint_files": sorted(p.name for p in trainer.checkpoint_dir.iterdir() if p.suffix == ".pt"),
    }
    shutil.rmtree(trainer.checkpoint_dir)
    return state_dict, history


def load_json(path):
    """Inputs: path (str | Path). Output: parsed JSON"""
    with open(path) as f:
        return json.load(f)


def save_json(obj, path):
    """Inputs: obj (JSON-serializable), path (str | Path). Output: none"""
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
