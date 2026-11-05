"""Mixed-batch cotraining: every source is shuffled into the same batches (as PushVIBES/train.py did
before Jul 7 2026), with each sample's loss scaled by its source's weight. Used by train.py
(mixed_batches=True). Copy of PushVIBES/mixed_batches.py (separate repo); keep the two in sync."""

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler


class SourceWeightedDataset(Dataset):
    """Wraps one source's dataset and adds its loss weight to every item.
    Inputs: dataset (Dataset of dict items), weight (float).
    Item: the wrapped item (dict) plus 'source_weight' (float32 scalar tensor, collated to [B])"""

    def __init__(self, dataset: Dataset, weight: float):
        self.dataset = dataset
        self.weight = torch.tensor(weight, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.dataset[idx]
        item["source_weight"] = self.weight
        return item


class MixedSourceSampler(Sampler):
    """Each epoch (each iteration) draws a fresh subset per source, without replacement, and
    shuffles all of them together, over a ConcatDataset of the sources in the same order.
    Inputs: source_lengths (list[int], len of each source), pts_per_epoch (list[int], points per
    source per epoch; None, < 0 or >= len uses the whole source). Yields: int indices into the ConcatDataset"""

    def __init__(self, source_lengths: list[int], pts_per_epoch: list[int]):
        self.source_lengths = source_lengths
        self.pts_per_epoch = [
            length if pts is None or pts < 0 or pts >= length else pts
            for length, pts in zip(source_lengths, pts_per_epoch)
        ]
        self.offsets = [sum(source_lengths[:i]) for i in range(len(source_lengths))]

    def __len__(self) -> int:
        return sum(self.pts_per_epoch)

    def __iter__(self):
        indices = torch.cat([
            offset + torch.randperm(length)[:pts]
            for offset, length, pts in zip(self.offsets, self.source_lengths, self.pts_per_epoch)
        ])
        return iter(indices[torch.randperm(len(indices))].tolist())


def build_mixed_train_loader(datasets: list[Dataset], weights: list[float], pts_per_epoch: list[int],
                             batch_size: int) -> DataLoader:
    """One DataLoader over all sources, shuffled together. Sources with pts_per_epoch 0 are skipped.
    Inputs: datasets (list[Dataset]), weights (list[float], per-source loss weight), pts_per_epoch (list[int]),
    batch_size (int). Output: DataLoader whose batches carry 'source_weight' ([B])"""
    kept = [(dataset, weight, pts) for dataset, weight, pts in zip(datasets, weights, pts_per_epoch) if pts != 0]
    if not kept:
        raise ValueError("No training data selected. Set at least one data_pts_per_epoch entry to nonzero.")

    dataset = ConcatDataset([SourceWeightedDataset(dataset, weight) for dataset, weight, _ in kept])
    sampler = MixedSourceSampler([len(dataset) for dataset, _, _ in kept], [pts for _, _, pts in kept])
    # One loader reused every epoch (the sampler redraws each iteration), so persistent workers aren't respawned
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=4, persistent_workers=True)


def weighted_mse(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """MSE with each sample's squared error scaled by its weight (equals nn.MSELoss() when all weights are 1).
    Inputs: pred, target ([B, 3]), weights ([B]). Output: scalar tensor"""
    return (weights.unsqueeze(1) * (pred - target) ** 2).mean()
