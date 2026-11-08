"""Plugs a brain into the simulation so agents act with it instead of randomly."""

import numpy as np
import torch

from src.observe import observe
from src.brain import MemoryBank


def to_tensors(view, self_feats, device):
    return (torch.from_numpy(view).to(device, non_blocking=True),
            torch.from_numpy(self_feats).to(device, non_blocking=True))


class BrainPolicy:
    def __init__(self, brain, device):
        self.brain = brain
        self.device = device
        self.sim = None
        self.bank = None

    def __call__(self, sim):
        if sim is not self.sim:              # a new world: everyone starts with an empty memory
            self.sim = sim
            self.bank = MemoryBank(self.brain, sim.agents.count, self.device)
        actions = np.zeros(sim.agents.count, dtype=np.int64)
        idx = np.flatnonzero(sim.agents.alive)
        if len(idx):
            view, feats = observe(sim, idx)
            memory = self.bank.get(sim.agents, idx)
            acts, _, _, memory = self.brain.act(*to_tensors(view, feats, self.device), memory)
            self.bank.set(idx, memory)
            actions[idx] = acts.cpu().numpy()
        return actions
