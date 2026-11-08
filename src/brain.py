"""The brain: one neural network shared by every agent.

    view (47 x 15 x 15) --> [EYES: 3 convolution layers] --\
                                                             >-> [THINKING: 2 layers] --+--> ACTOR  -> 19 action chances
    self (94 numbers)   --> [BODY SENSE: 1 layer] ----------/            |              |
                                                                         v              +--> CRITIC -> "how good is my situation?"
                                                   MEMORY (last tick) -> [MEMORY: GRU] -+
                                                                         |
                                                                         +--> MEMORY (next tick)

EYES: convolution layers slide small 3 x 3 "pattern detectors" over the view.
      Each detector learns to light up for one useful pattern, such as
      "water right next to me" or "something with glowing eyes nearby".
BODY SENSE: turns the self numbers into features like "very thirsty".
THINKING: combines what it sees with how it feels, right now.
MEMORY (added in Stage 3): a GRU, a small layer that keeps 128 numbers from one
      tick to the next. Each tick it decides what to keep, what to forget and
      what to write, so an agent can remember things it can no longer see
      ("there was water behind that hill", "a wolf chased me from the east").
      Each agent has its own memory; a newborn starts with an empty one.
ACTOR: a score for each action, turned into chances (softmax).
CRITIC: one number, its guess of how much future reward to expect.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from src.observe import NUM_CHANNELS, NUM_SELF, VIEW_SIZE
from src.agents import NUM_ACTIONS

MEMORY = 128     # how many numbers each agent remembers between ticks


class Brain(nn.Module):
    """channels / self_size / actions / memory default to the current world.
    Older brains can be rebuilt by passing their sizes (memory=0: no memory),
    which is what brain surgery (surgery.py) does."""

    def __init__(self, hidden=256, channels=NUM_CHANNELS, self_size=NUM_SELF,
                 actions=NUM_ACTIONS, memory=MEMORY):
        super().__init__()
        self.sizes = {"hidden": hidden, "channels": channels, "self_size": self_size,
                      "actions": actions, "memory": memory}
        self.memory_size = memory
        self.eyes = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1), nn.ReLU(),                # 15 x 15
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),            # 8 x 8
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),            # 4 x 4
            nn.Flatten(),
        )
        with torch.no_grad():
            eye_size = self.eyes(torch.zeros(1, channels, VIEW_SIZE, VIEW_SIZE)).shape[1]
        self.body = nn.Sequential(nn.Linear(self_size, 64), nn.ReLU())
        self.thinking = nn.Sequential(
            nn.Linear(eye_size + 64, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        if memory:
            self.memory = nn.GRUCell(hidden, memory)
        self.actor = nn.Linear(hidden + memory, actions)
        self.critic = nn.Linear(hidden + memory, 1)

        # Start the actor almost neutral (every action about equally likely),
        # so the newborn brain explores everything
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)

    # ----- the pieces -----
    def think(self, view, self_features):
        """What the agent makes of this tick (no memory involved)."""
        return self.thinking(torch.cat([self.eyes(view), self.body(self_features)], dim=1))

    def remember(self, thought, memory):
        """Update the memory with this tick's thought."""
        return self.memory(thought, memory) if self.memory_size else memory

    def decide(self, thought, memory):
        z = torch.cat([thought, memory], dim=1) if self.memory_size else thought
        return self.actor(z), self.critic(z).squeeze(1)

    def empty_memory(self, n, device):
        return torch.zeros(n, self.memory_size, device=device)

    # ----- one tick -----
    def forward(self, view, self_features, memory=None):
        """Returns (action scores, value guess, new memory)."""
        if memory is None:
            memory = self.empty_memory(view.shape[0], view.device)
        thought = self.think(view, self_features)
        memory = self.remember(thought, memory)
        logits, value = self.decide(thought, memory)
        return logits, value, memory

    @torch.no_grad()
    def act(self, view, self_features, memory=None, greedy=False):
        """Pick an action for each agent. Returns (actions, log_probs, values, new memory).
        greedy=True always picks the most likely action (used only for testing)."""
        logits, values, memory = self(view, self_features, memory)
        dist = Categorical(logits=logits)
        actions = logits.argmax(dim=1) if greedy else dist.sample()
        return actions, dist.log_prob(actions), values, memory

    # ----- learning: replay short stretches of an agent's life -----
    def evaluate_sequences(self, view, self_features, actions, start_memory, layout):
        """Used while learning. The decisions come in short stretches of one
        agent's life (layout: stretches x SEQ_LEN, each entry the row of a
        decision, or -1 for padding). The memory is replayed through each
        stretch so the brain learns what was worth remembering.
        Returns, for every decision (in row order): how likely the brain NOW
        finds the action that was taken, how unsure it is (entropy), and its
        value guess."""
        n = view.shape[0]
        thought = self.think(view, self_features)                 # (n, hidden)
        stretches, length = layout.shape
        valid = layout >= 0
        rows = layout.clamp(min=0)
        memory = start_memory
        out_logits = torch.zeros(n, self.actor.out_features, device=view.device)
        out_values = torch.zeros(n, device=view.device)
        for t in range(length):
            here = valid[:, t]
            if not here.any():
                break
            th = thought[rows[:, t]]
            new_memory = self.remember(th, memory)
            memory = torch.where(here[:, None], new_memory, memory)
            logits, value = self.decide(th, memory)
            r = rows[here, t]
            out_logits[r] = logits[here]
            out_values[r] = value[here]
        dist = Categorical(logits=out_logits)
        return dist.log_prob(actions), dist.entropy(), out_values


class MemoryBank:
    """Every agent's memory, kept between ticks (one row per agent slot).
    Slots are reused after deaths, so a slot's memory is wiped whenever a
    new person moves in (recognised by their spawn tick)."""

    def __init__(self, brain, capacity, device):
        self.device = device
        self.memory = brain.empty_memory(capacity, device)
        self.owner = np.full(capacity, -1, dtype=np.int64)

    def get(self, agents, idx):
        idx = np.asarray(idx)
        new = agents.spawn_tick[idx] != self.owner[idx]
        if new.any():
            rows = idx[new]
            self.memory[torch.from_numpy(rows).to(self.device)] = 0
            self.owner[rows] = agents.spawn_tick[rows]
        return self.memory[torch.from_numpy(idx).to(self.device)]

    def set(self, idx, memory):
        self.memory[torch.from_numpy(np.asarray(idx)).to(self.device)] = memory


def count_weights(model):
    return sum(p.numel() for p in model.parameters())


def save_brain(brain, path, extra=None):
    torch.save({"weights": brain.state_dict(), "sizes": brain.sizes, "extra": extra or {}}, path)


def load_weights(brain, data, path=""):
    """Put saved weights into a brain, with a clear message if they don't fit."""
    try:
        brain.load_state_dict(data["weights"])
    except RuntimeError:
        raise SystemExit(
            f"The brain in {path} was made for an older version of the world (different senses,\n"
            f"actions or memory). Upgrade it first with brain surgery, for example:\n"
            f"    python surgery.py {path or 'runs/run1/latest.pt'} --name run2")


def load_brain(path, device="cpu"):
    data = torch.load(path, map_location=device, weights_only=False)
    brain = Brain().to(device)
    load_weights(brain, data, path)
    return brain, data.get("extra", {})
