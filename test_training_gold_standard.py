"""Gold-standard training test: run after every change to make sure training is reproducible.

Builds PushVibesTrainer (train.py) the way main() does (set_seed, spawned DataLoader workers,
sim + real cotraining with per-source weights), starting from frozen initial weights
(unit_test_data/training/initial_weights.pt), trains 5 tiny epochs on frozen 512-point
episodes, and checks the resulting weights and loss histories are IDENTICAL to the recorded run.
One test per case in training_gold_standard_common.TRAINING_CASES.

Run from the repo root (needs CUDA):
    python -m unittest test_training_gold_standard -v

A failure means training behavior changed. Only regenerate the gold standard
(generate_training_gold_standard_data.py) when you WANT to move it.
"""

import unittest

import torch

from training_gold_standard_common import TRAINING_CASES, check_environment, environment_info, run_training, load_json


def diff_state_dicts(actual, expected):
    """Inputs: actual, expected (dict[str, torch.Tensor]).
    Output: list[str] describing every tensor that is missing, extra, reshaped, or not bitwise equal"""
    failures = [f"missing: {k}" for k in expected if k not in actual]
    failures += [f"unexpected: {k}" for k in actual if k not in expected]
    for key in expected:
        if key not in actual:
            continue
        a, e = actual[key], expected[key]
        if a.shape != e.shape or a.dtype != e.dtype:
            failures.append(f"{key}: {a.dtype}{tuple(a.shape)} != expected {e.dtype}{tuple(e.shape)}")
        elif not torch.equal(a, e):
            failures.append(f"{key}: max_abs_diff={(a.double() - e.double()).abs().max().item():.3g}")
    return failures


class TestTrainingGoldStandard(unittest.TestCase):

    def check_training(self, case):
        """Train one TRAINING_CASES case and compare it bitwise to its gold standard.
        Inputs: case (str, key of TRAINING_CASES). Output: none"""
        check_environment()
        trainer_kwargs, weights_path, history_path = TRAINING_CASES[case]
        expected = load_json(history_path)
        actual_weights, actual_history = run_training(trainer_kwargs)

        failures = diff_state_dicts(actual_weights, torch.load(weights_path, map_location="cpu"))
        # Histories localize when the drift started (which epoch, train vs val)
        for name, expected_history in expected["history"].items():
            if actual_history[name] != expected_history:
                failures.append(f"history '{name}': got={actual_history[name]} expected={expected_history}")
        if failures:
            note = "" if expected["environment"] == environment_info() else (
                f"\nNOTE: environment differs from gold standard: recorded={expected['environment']} "
                f"current={environment_info()}")
            self.fail(f"Training ({case}) drifted from gold standard:\n  " + "\n  ".join(failures) + note)

    def test_end_point_trained_weights_match(self):
        self.check_training("end_point")

    def test_displacement_trained_weights_match(self):
        self.check_training("displacement")

    def test_end_point_mixed_batches_trained_weights_match(self):
        self.check_training("end_point_mixed_batches")


if __name__ == "__main__":
    unittest.main()
