import torch
import torch.optim as optim
import torch.nn as nn
import random
import numpy as np
import os
import logging
import json
from datetime import datetime
from pathlib import Path
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
import itertools
from tqdm import tqdm
from PushVIB3S_start_point_conditioned import PushVIB3S_start_point_conditioned
from dataset import PushDataset
import pickle
import torch.multiprocessing as mp
from typing import Dict, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils import set_seed, save_val_error_plot, save_val_loss_plot

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _append_loss_history(history: Dict[str, list], losses: Dict[str, float]) -> None:
    """Append one epoch's loss components to history. The "kl" line is the effective
    (beta_goal-scaled) KL loss, i.e. its actual contribution to total_loss."""
    history["total"].append(losses["total_loss"])
    history["end_point"].append(losses["end_point_loss"])
    history["kl"].append(losses["kl_loss_effective"])


class PushVibesTrainer:
    def __init__(
        self,
        *,
        n_neurons: int,
        in_bps: int,
        goalD: int,
        feature_dim: int,
        beta_goal: float,
        batch_size: int,
        learning_rate: float,
        num_epochs: int,
        seed: int,
        deterministic: bool = False,
        use_directional_bps: bool = True,
        sim_validation: bool = True,
        sim_data_dir: str = None,
        real_data_dir: str = None,
        validation_data_dir: str = None,
        pretrained_weights_path: str = None,
        sim_data_pts_per_epoch: int = -1, # -1 means use all data, otherwise specify how many points to use per epoch
        real_data_pts_per_epoch: int = -1, # -1 means use all data, otherwise specify how many points to use per epoch
        real_loss_weight: float = 1.0,     # scale real-data loss relative to sim (>1.0 emphasizes real)
        run_name_prefix: str = "",
    ):
        self.n_neurons = n_neurons
        self.in_bps = in_bps
        self.goalD = goalD
        self.feature_dim = feature_dim
        self.beta_goal_after_a_few_epochs = beta_goal
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.seed = seed
        self.deterministic = deterministic
        self.use_directional_bps = use_directional_bps
        self.sim_validation = sim_validation
        self.sim_data_dir = sim_data_dir
        self.real_data_dir = real_data_dir
        self.validation_data_dir = validation_data_dir
        self.pretrained_weights_path = pretrained_weights_path
        self.sim_data_pts_per_epoch = sim_data_pts_per_epoch
        self.real_data_pts_per_epoch = real_data_pts_per_epoch
        self.real_loss_weight = real_loss_weight
        self.device = torch.device('cuda')
        self.run_name_prefix = run_name_prefix
        self.run_name = self._get_run_name()
        checkpoint_root = Path('checkpoints') / 'deterministic' if self.deterministic else Path('checkpoints')
        self.checkpoint_dir = checkpoint_root / self.run_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.current_epoch = 1
        self.sim_val_error_history = {"total": [], "end_point": []}
        self.real_val_error_history = {"total": [], "end_point": []}
        self.train_loss_history = {"total": [], "end_point": [], "kl": []}
        self.sim_val_loss_history = {"total": [], "end_point": [], "kl": []}
        self.real_val_loss_history = {"total": [], "end_point": [], "kl": []}
        self._save_run_config()

        self._init_dataset()
        self._setup()
        self._load_pretrained_weights()
        self._init_optimizer()

    def _setup(self) -> None:

        self.model = PushVIB3S_start_point_conditioned(
            n_neurons=self.n_neurons,
            in_bps=self.in_bps,  # This is the base BPS dimension
            goalD=self.goalD,
            feature_dim=self.feature_dim,
            deterministic=self.deterministic,
            use_directional_bps=True  # Explicitly set to use directional BPS
        ).to(self.device)
        self.l2_loss = nn.MSELoss()

    def _init_optimizer(self) -> None:
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate
        )

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=50,
        )

    def _init_dataset(self) -> None:

        # Split sim dataset into train and validation
        full_dataset = PushDataset(
            data_dir=self.sim_data_dir,
            use_directional_bps=True  # Set to True to use directional BPS
        )

        # Use 90% for training, 10% for validation
        train_size = int(0.9 * len(full_dataset))
        val_size = len(full_dataset) - train_size

        self.sim_train_dataset, self.sim_val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )

        if self.sim_validation:
            # Subsample sim val dataset to num_sim_val_pts
            num_sim_val_pts = 2000
            if num_sim_val_pts < len(self.sim_val_dataset):
                self.sim_val_dataset = Subset(self.sim_val_dataset, torch.randperm(len(self.sim_val_dataset))[:num_sim_val_pts].tolist())

            self.sim_val_loader = DataLoader(
                self.sim_val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=4,
                persistent_workers=True
            )

        # if self.pretrained_weights_path is not None or self.sim_data_pts_per_epoch == 0:
        self.real_train_dataset = PushDataset(
            data_dir=self.real_data_dir,
            use_directional_bps=True
        )

        self.real_val_dataset = PushDataset(
            data_dir=self.validation_data_dir,
            use_directional_bps=True,
            deterministic_farthest_point_sampling=True  # Ensure deterministic sampling for validation
        )

        self.real_val_loader = DataLoader(
            self.real_val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            persistent_workers=True
        )

        # Built once and reused every epoch so persistent workers don't get respawned
        # each epoch.
        self.train_loaders: list[tuple[DataLoader, float]] = []
        if self.sim_data_pts_per_epoch != 0:
            self.train_loaders.append((self._build_train_loader(self.sim_train_dataset, self.sim_data_pts_per_epoch), 1.0))
        if self.real_data_pts_per_epoch != 0:
            self.train_loaders.append((self._build_train_loader(self.real_train_dataset, self.real_data_pts_per_epoch), self.real_loss_weight))

        if not self.train_loaders:
            raise ValueError("No training data selected. Set sim_data_pts_per_epoch or real_data_pts_per_epoch > 0.")

    def _build_train_loader(self, dataset: torch.utils.data.Dataset, pts_per_epoch: int) -> DataLoader:
        # Both shuffle=True and RandomSampler re-draw fresh indices every time the loader
        # is iterated, so a single DataLoader (and its persistent workers) can be reused
        # across epochs instead of rebuilt each epoch.
        if pts_per_epoch is None or pts_per_epoch < 0 or pts_per_epoch >= len(dataset):
            print(f"Using the entire dataset of size {len(dataset)} every epoch.")
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, persistent_workers=True)

        # Subsampling k < len(dataset) points per epoch requires sampling with replacement
        # (RandomSampler only supports num_samples != len(dataset) that way); duplicates
        # within one epoch's subset are possible but very unlikely to matter for training.
        sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=pts_per_epoch)
        return DataLoader(dataset, batch_size=self.batch_size, sampler=sampler, num_workers=4, persistent_workers=True)

    def _get_run_name(self) -> str:
        job_id = os.getenv('SLURM_JOB_ID', 'local')
        prefix = self.run_name_prefix
        return f"{prefix}{job_id}_{datetime.now():%Y%m%d_%H%M%S}"

    def _load_pretrained_weights(self) -> None:
        if not self.pretrained_weights_path:
            return

        if not os.path.exists(self.pretrained_weights_path):
            logger.warning(
                "Pretrained weights not found at %s; continuing with random init.",
                self.pretrained_weights_path,
            )
            return

        params_before = sum(p.detach().abs().sum().item() for p in self.model.parameters())
        loaded = torch.load(self.pretrained_weights_path, map_location=self.device)
        state_dict = loaded.get("model_state", loaded)
        self.model.load_state_dict(state_dict, strict=True)
        params_after = sum(p.detach().abs().sum().item() for p in self.model.parameters())
        logger.info(
            "Loaded pretrained weights from %s (param_abs_sum %.6f -> %.6f)",
            self.pretrained_weights_path,
            params_before,
            params_after,
        )

        if params_before == params_after:
            logger.warning(
                "The sum of absolute parameter values is unchanged after loading pretrained weights."
            )


    def _save_run_config(self) -> None:
        config = {
            "n_neurons": self.n_neurons,
            "in_bps": self.in_bps,
            "goalD": self.goalD,
            "feature_dim": self.feature_dim,
            "beta_goal": self.beta_goal_after_a_few_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "seed": self.seed,
            "deterministic": self.deterministic,
            "use_directional_bps": self.use_directional_bps,
            "sim_validation": self.sim_validation,
            "sim_data_dir": self.sim_data_dir,
            "real_data_dir": self.real_data_dir,
            "validation_data_dir": self.validation_data_dir,
            "pretrained_weights_path": self.pretrained_weights_path,
            "sim_data_pts_per_epoch": self.sim_data_pts_per_epoch,
            "real_data_pts_per_epoch": self.real_data_pts_per_epoch,
            "real_loss_weight": self.real_loss_weight,
        }

        out_path = self.checkpoint_dir / "a_run_config.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, sort_keys=True)


    def save_checkpoint(self, epoch: int, loss: float, is_best: bool) -> Path | None:
        path = self.checkpoint_dir / f'checkpoint_{epoch}_{loss:.6f}.pt'

        # Always save the latest weights (overwriting each time)
        torch.save(self.model.state_dict(), self.checkpoint_dir / 'latest_model_weights.pt')

        if is_best:
            # Save the best weights using the epoch/loss path
            torch.save(self.model.state_dict(), path)
            return path

        return None


    def _compute_losses(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Compute (unweighted) loss tensors for one batch, without stepping the optimizer."""
        start_point_normalized = batch['start_point_normalized'].squeeze(1)  # From [B,1,3] to [B,3]

        results = self.model(
            batch['start_bps'],
            batch['goal_bps'],
            start_point_normalized,
            x_mean=batch['x_mean'],  # Pass the mean for denormalization
            x_max=batch['x_max']       # Pass the max for denormalization
        )

        # Loss is computed in normalized space
        pred_end_point = results['normalized_end_point']
        target_end_point = batch['end_point_normalized'].squeeze(1)  # From [B,1,3] to [B,3]

        # Reconstruction loss for the predicted end point
        end_point_loss = self.l2_loss(pred_end_point, target_end_point)

        # KL divergence losses
        kl_loss = results['kl_goal']
        effective_kl_loss = self.beta_goal * kl_loss  # KL loss as actually weighted into total_loss

        total_loss = end_point_loss + effective_kl_loss

        return {
            'total_loss': total_loss,
            'end_point_loss': end_point_loss,
            'kl_loss': kl_loss,
            'kl_loss_effective': effective_kl_loss
        }

    def train_step(self, batches: list[Dict[str, Any]], weights: list[float]) -> Dict[str, float]:
        """Combine one batch per data source into a single weighted loss and take one
        optimizer step, so cotraining sources share a gradient update instead of each
        source getting its own separate step."""
        per_source_losses = [self._compute_losses(batch) for batch in batches]

        combined_loss = sum(
            weight * losses['total_loss'] for weight, losses in zip(weights, per_source_losses)
        )

        self.optimizer.zero_grad()
        combined_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # Every component is combined with the same per-source weighting used to build
        # combined_loss, so total_loss == end_point_loss + kl_loss_effective always holds
        # (rather than mixing a weighted sum for total_loss with a plain average for the
        # per-component metrics, which only coincidentally match with one active source).
        avg_losses = {
            k: sum(weight * losses[k].detach().item() for weight, losses in zip(weights, per_source_losses))
            for k in per_source_losses[0]
        }
        return avg_losses

    def _run_batches(self, loader, step_fn) -> Dict[str, float]:
        """Run step_fn over every batch in loader, returning the per-key average across batches."""
        totals: Dict[str, float] = {}
        n_batches = 0

        for batch in loader:
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                    for k, v in batch.items()}

            losses = step_fn(batch)
            for k, v in losses.items():
                totals[k] = totals.get(k, 0.0) + v
            n_batches += 1

            if hasattr(loader, "set_postfix"):
                loader.set_postfix({k: f'{v:.4f}' for k, v in losses.items()})

        return {k: v / n_batches for k, v in totals.items()}

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()

        loaders = [loader for loader, _ in self.train_loaders]
        weights = [weight for _, weight in self.train_loaders]

        # Step once per batch of the longest loader; shorter loaders are cycled so every
        # step has a batch from every source.
        steps_per_epoch = max(len(loader) for loader in loaders)
        iters = [
            iter(loader) if len(loader) == steps_per_epoch else itertools.cycle(loader)
            for loader in loaders
        ]

        totals: Dict[str, float] = {}
        pbar = tqdm(total=steps_per_epoch, desc=f'Epoch {epoch}')

        for _ in range(steps_per_epoch):
            batches = [
                {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
                for batch in (next(it) for it in iters)
            ]
            losses = self.train_step(batches, weights)
            for k, v in losses.items():
                totals[k] = totals.get(k, 0.0) + v
            pbar.set_postfix({k: f'{v:.4f}' for k, v in losses.items()})
            pbar.update(1)

        pbar.close()
        return {k: v / steps_per_epoch for k, v in totals.items()}

    def validation_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        start_point_normalized = batch['start_point_normalized'].squeeze(1)

        results = self.model(
            batch['start_bps'],
            batch['goal_bps'],
            start_point_normalized,
            x_mean=batch['x_mean'],  # Pass the mean for denormalization
            x_max=batch['x_max']       # Pass the max for denormalization
        )

        # Loss is computed in normalized space, to match train_step
        pred_end_point_normalized = results['normalized_end_point']
        target_end_point_normalized = batch['end_point_normalized'].squeeze(1)

        # KL divergence loss
        kl_loss = results['kl_goal']
        effective_kl_loss = self.beta_goal * kl_loss  # KL loss as actually weighted into total_loss

        # Reconstruction loss
        end_point_loss = self.l2_loss(target_end_point_normalized, pred_end_point_normalized)

        total_loss = end_point_loss + effective_kl_loss

        # Error metric stays in real-world units for interpretable logging/early stopping.
        # Per-sample Euclidean distance, averaged over the batch (not a single Frobenius
        # norm over the whole batch tensor).
        pred_end_point = results['end_point']
        target_end_point = batch['end_point'].squeeze(1)

        end_point_error = torch.norm(pred_end_point - target_end_point, dim=1).mean().item()

        return {
            'total_loss': total_loss.item(),
            'end_point_loss': end_point_loss.item(),
            'kl_loss': kl_loss.item(),
            'kl_loss_effective': effective_kl_loss.item(),
            'end_point_error': end_point_error,
        }

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        with torch.no_grad():
            return self._run_batches(val_loader, self.validation_step)

    def _record_validation(self, source: str, losses: Dict[str, float]) -> float:
        """Append one epoch's error/loss history for 'sim' or 'real' validation, return the total error."""
        error_history = self.sim_val_error_history if source == "sim" else self.real_val_error_history
        loss_history = self.sim_val_loss_history if source == "sim" else self.real_val_loss_history

        total_error = losses["end_point_error"]
        error_history["total"].append(total_error)
        error_history["end_point"].append(losses["end_point_error"])

        _append_loss_history(loss_history, losses)

        return total_error

    def _log_validation(self, label: str, source: str, losses: Dict[str, float]) -> None:
        logger.info(
            f'{label}: '
            f'{source}_val_loss={losses["total_loss"]:.4f}, '
            f'{source}_val_end_point_loss={losses["end_point_loss"]:.4f}, '
            f'{source}_val_kl_loss={losses["kl_loss"]:.4f}'
        )

    def evaluate_current_weights(self, label: str) -> Dict[str, Any]:
        sim_val_losses = None
        sim_total_error = None

        if self.sim_validation:
            sim_val_losses = self.validate(self.sim_val_loader)
            sim_total_error = self._record_validation("sim", sim_val_losses)
            self._log_validation(label, "sim", sim_val_losses)

        real_val_losses = self.validate(self.real_val_loader)
        real_total_error = self._record_validation("real", real_val_losses)
        self._log_validation(label, "real", real_val_losses)

        return {
            "sim": sim_val_losses,
            "real": real_val_losses,
            "sim_total_error": sim_total_error,
            "real_total_error": real_total_error,
        }

    def _checkpoint_metrics(self, eval_results: Dict[str, Any], use_sim_for_best: bool) -> tuple[float, float]:
        """Return (loss used to track the best epoch, loss used in the checkpoint filename)."""
        if use_sim_for_best:
            return eval_results["sim"]["total_loss"], eval_results["sim_total_error"]
        return eval_results["real"]["total_loss"], eval_results["real_total_error"]

    def train(self) -> None:
        patience = 150
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        min_delta = 1e-6
        use_sim_for_best = self.sim_validation and self.sim_data_pts_per_epoch == -1 and self.real_data_pts_per_epoch == 0
        # beta_goal is 0 during these epochs (see the schedule below), so their val loss has
        # no KL term and isn't comparable to post-warmup losses. Best-weight tracking is
        # deferred until beta_goal reaches its target value, so an artificially low
        # pre-KL loss can never lock out every epoch that follows it.
        kl_warmup_epochs = 3

        self.beta_goal = 0.0
        initial_eval = self.evaluate_current_weights("Initial evaluation")
        _initial_best_val_loss, initial_checkpoint_loss = self._checkpoint_metrics(initial_eval, use_sim_for_best)
        self.scheduler.step(initial_eval["real"]["total_loss"])

        # Pre-training, beta_goal=0 eval is never eligible to be "best" for the same reason.
        self.save_checkpoint(-1, initial_checkpoint_loss, False)

        save_val_error_plot(
            self.sim_val_error_history,
            self.real_val_error_history,
            str(self.checkpoint_dir),
            -1,
        )
        save_val_loss_plot(
            self.sim_val_loss_history,
            self.real_val_loss_history,
            str(self.checkpoint_dir),
            -1,
        )

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch

            if epoch < kl_warmup_epochs:
                self.beta_goal = 0.0
            else:
                self.beta_goal = self.beta_goal_after_a_few_epochs # 0.00001 for sim training only, Why is this so low?

            # Training
            train_losses = self.train_epoch(epoch)
            _append_loss_history(self.train_loss_history, train_losses)

            # Validation
            eval_results = self.evaluate_current_weights(f"Epoch {epoch}")
            best_epoch_val_loss, checkpoint_loss = self._checkpoint_metrics(eval_results, use_sim_for_best)

            # Scheduler step with validation loss
            self.scheduler.step(eval_results["real"]["total_loss"])

            # Check for improvement. Epochs before the KL term is fully active are never
            # eligible to be "best" (see kl_warmup_epochs above); the first post-warmup
            # epoch unconditionally establishes the baseline instead of competing against
            # a pre-KL loss it structurally can't beat.
            if epoch < kl_warmup_epochs:
                is_best = False
            elif epoch == kl_warmup_epochs:
                is_best = True
            else:
                is_best = best_epoch_val_loss < (best_val_loss - min_delta)

            if is_best:
                best_val_loss = best_epoch_val_loss
                epochs_without_improvement = 0
            elif epoch >= kl_warmup_epochs:
                epochs_without_improvement += 1

            self.save_checkpoint(epoch, checkpoint_loss, is_best)

            logger.info(f'Epoch {epoch}: train_loss={train_losses["total_loss"]:.4f}')

            # Early stopping check
            if epochs_without_improvement >= patience:
                logger.info(f'Early stopping triggered after {epoch} epochs')
                break

            save_val_error_plot(
                self.sim_val_error_history,
                self.real_val_error_history,
                str(self.checkpoint_dir),
                epoch,
            )
            save_val_loss_plot(
                self.sim_val_loss_history,
                self.real_val_loss_history,
                str(self.checkpoint_dir),
                epoch,
            )
            train_epochs = list(range(1, len(self.train_loss_history["total"]) + 1))
            plt.figure(figsize=(10, 6))
            plt.plot(train_epochs, self.train_loss_history["total"], linestyle="-", label="total")
            plt.plot(train_epochs, self.train_loss_history["end_point"], linestyle="--", label="end point")
            plt.plot(train_epochs, self.train_loss_history["kl"], linestyle="-.", label="kl (effective)")
            plt.xlabel("Epoch")
            plt.ylabel("Training Loss")
            plt.title("Training Loss Curves")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(str(self.checkpoint_dir / "train_loss_curve.png"))
            plt.close()

def main() -> None:
    mp.set_start_method('spawn', force=True)
    # Set random seed
    set_seed(333)

    run_name_prefix = "realsense_bg0.08"
    if run_name_prefix is None:
        run_name_prefix = input("Enter a run name prefix (or leave blank for none): ").strip()

    # Initialize trainer and start training
    trainer = PushVibesTrainer(
        n_neurons=512,
        in_bps=128,
        goalD=7,  # 2, 7
        feature_dim=128,
        beta_goal=0.08, # 0.00001 was used for sim only training
        batch_size=64, #fine tuning does better with 64 than 200  # 512, 2048 was used for sim training
        learning_rate=5e-5, # 1e-4 was used for sim training, 1e-4 cotrain, 1e-5 / 8e-5 was used for finetuning
        num_epochs=600, # 350 usually sufficient for fine/co-training
        seed=334,
        deterministic=False,
        use_directional_bps=True,
        sim_validation=False,  # Set to False to skip sim validation loss/plot
        sim_data_dir='/home/britton/PushVIBES/data/aug7_2025_new_data/processed_data_noisy_object_frame_with_old_and_new_data',
        real_data_dir='/home/britton/PushVIBES/data/all_good_realsense_data/training',
        validation_data_dir='/home/britton/PushVIBES/data/all_good_realsense_data/test',
        # pretrained_weights_path='./weights/oriented/new_old_data_mix_checkpoint_noisy_115_0.000462.pt',
        # pretrained_weights_path='checkpoints/deterministic/deterministic_all_simlocal_20260407_161509/latest_model_weights.pt',
        pretrained_weights_path=None, # Set to None to train from scratch
        sim_data_pts_per_epoch=612, # 561
        real_data_pts_per_epoch=-1, # 561, -1 is all
        real_loss_weight=1.0,
        run_name_prefix=run_name_prefix
    )
    trainer.train()

    # Save the model's state_dict (weights) to a pickle file
    pickle_path = trainer.checkpoint_dir / 'pushVIBES_model_weights.pkl'
    pickle.dump(trainer.model.state_dict(), open(pickle_path, 'wb'))


if __name__ == "__main__":
    main()
