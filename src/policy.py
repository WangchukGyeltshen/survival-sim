"""Plugs a brain into the simulation so agents act with it instead of randomly."""

import numpy as np
import torch

from src.observe import observe


def to_tensors(view, self_feats, device):
    return (torch.from_numpy(view).to(device, non_blocking=True),
            torch.from_numpy(self_feats).to(device, non_blocking=True))


class BrainPolicy:
    def __init__(self, brain, device):
        self.brain = brain
        self.device = device

    def __call__(self, sim):
        actions = np.zeros(sim.agents.count, dtype=np.int64)
        idx = np.flatnonzero(sim.agents.alive)
        if len(idx):
            view, feats = observe(sim, idx)
            acts, _, _ = self.brain.act(*to_tensors(view, feats, self.device))
            actions[idx] = acts.cpu().numpy()
        return actions
