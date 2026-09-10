"""Gold-standard test: run after every change to make sure predictions haven't drifted.

Loads a fixed checkpoint into PushVIB3S_start_point_conditioned via DeployPushVIB3S.predict_push
(deploy.py) and checks its output on 3 frozen real-world data points (unit_test_data/) against
the "gold standard" predictions recorded in unit_test_data/gold_standard_predictions.json.

This catches two kinds of breakage:
  1. The checkpoint no longer loads into the current model architecture at all (shape
     mismatch in load_state_dict -> every test in this file errors out).
  2. The checkpoint loads fine, but predictions changed -- i.e. something in the
     preprocessing/model/deploy pipeline silently changed behavior even though the weights
     didn't.

Run with:
    python test_deploy_gold_standard.py
    python -m unittest test_deploy_gold_standard -v

If you INTENTIONALLY changed something that should change these predictions (retrained,
deliberately changed preprocessing, etc.), regenerate the gold standard with:
    python generate_gold_standard_data.py
and review the new predicted_end_point/predicted_displacement values before committing them.
"""

from pathlib import Path
import json
import unittest

import numpy as np

from deploy import DeployPushVIB3S

# --- must match generate_gold_standard_data.py so the gold standard stays comparable ---
DATA_DIR = Path("unit_test_data")
GOLD_STANDARD_JSON_PATH = DATA_DIR / "gold_standard_predictions.json"
DEVICE = "cpu"  # fixed (not cuda) so results are reproducible on any machine
# Max allowed drift (meters) between a fresh prediction and the recorded gold standard.
# Same seed + same weights + same code should reproduce the gold standard almost exactly;
# this only needs to absorb harmless float noise (e.g. BLAS thread-count differences), not
# real prediction changes, which show up at the cm scale.
TOLERANCE_METERS = 1e-4
# -------------------------------------------------------------------------------------


class TestDeployMatchesGoldStandard(unittest.TestCase):
    """Locks in DeployPushVIB3S.predict_push's output for one checkpoint on 3 fixed inputs."""

    @classmethod
    def setUpClass(cls):
        if not GOLD_STANDARD_JSON_PATH.exists():
            raise FileNotFoundError(
                f"{GOLD_STANDARD_JSON_PATH} not found. Run generate_gold_standard_data.py first."
            )
        with open(GOLD_STANDARD_JSON_PATH) as f:
            cls.gold_standard = json.load(f)

        # Loading the checkpoint here means a shape/architecture mismatch fails every test
        # in this class with a clear load_state_dict error, instead of silently skipping.
        cls.deployer = DeployPushVIB3S(
            weights_path=cls.gold_standard["weights_path"],
            basis_path=cls.gold_standard["basis_path"],
            device=DEVICE,
            encoder_type=cls.gold_standard["encoder_type"],
        )

    def test_predictions_match_gold_standard(self):
        self.assertTrue(
            self.gold_standard["episodes"],
            "gold_standard_predictions.json has no recorded episodes",
        )

        failures = []
        for episode in self.gold_standard["episodes"]:
            fixture_path = DATA_DIR / episode["fixture_file"]
            fixture = np.load(fixture_path, allow_pickle=True)

            np.random.seed(episode["seed"])
            predicted_end_point, predicted_displacement = self.deployer.predict_push(
                fixture["start_pointcloud"],
                fixture["goal_pointcloud"],
                fixture["push_start_point"],
            )

            expected_end_point = np.asarray(episode["predicted_end_point"])
            expected_displacement = np.asarray(episode["predicted_displacement"])

            end_point_diff = np.abs(predicted_end_point - expected_end_point).max()
            displacement_diff = np.abs(predicted_displacement - expected_displacement).max()

            if end_point_diff > TOLERANCE_METERS or displacement_diff > TOLERANCE_METERS:
                failures.append(
                    f"  {episode['fixture_file']} ({episode['source_path']}):\n"
                    f"    end_point:    got={predicted_end_point} expected={expected_end_point} "
                    f"max_diff={end_point_diff:.6f}\n"
                    f"    displacement: got={predicted_displacement} "
                    f"expected={expected_displacement} max_diff={displacement_diff:.6f}"
                )

        if failures:
            self.fail(
                "Predictions drifted from the gold standard in "
                f"{GOLD_STANDARD_JSON_PATH} (tolerance={TOLERANCE_METERS} m):\n"
                + "\n".join(failures)
                + "\n\nIf this drift is expected (intentional retrain/behavior change), "
                "regenerate the gold standard with generate_gold_standard_data.py."
            )


if __name__ == "__main__":
    unittest.main()
