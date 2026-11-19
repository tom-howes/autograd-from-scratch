"""Smoke test: train a small network built from the hand-written operators on noisy XOR.

XOR is the right sanity check because it is not linearly separable, so the network can only
solve it if the non-linearity and its analytically derived gradient are both correct.
"""
import torch
from torch.utils.data import TensorDataset

from create_net import create_net
from train import train


def make_xor(n_per_quadrant=250, spread=0.25, seed=0):
    """Generate a noisy XOR dataset centred on the four unit-square corners."""
    g = torch.Generator().manual_seed(seed)
    centres = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    labels = torch.tensor([0, 1, 1, 0])

    xs, ys = [], []
    for centre, label in zip(centres, labels):
        pts = centre + spread * torch.randn(n_per_quadrant, 2, generator=g)
        xs.append(pts)
        ys.append(torch.full((n_per_quadrant,), int(label), dtype=torch.long))

    return TensorDataset(torch.cat(xs), torch.cat(ys))


if __name__ == "__main__":
    dataset = make_xor()

    net = create_net(
        input_features=2,
        hidden_units=[16, 16],
        non_linearity=["tanH", "tanH"],
        output_size=2,
    )

    train(net, dataset, {
        "num_epochs": 40,
        "lr": 0.1,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "step_size": 15,
        "gamma": 0.5,
    })
