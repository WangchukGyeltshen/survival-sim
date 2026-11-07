"""The brain: one neural network shared by every agent.

    view (27 x 15 x 15) --> [EYES: 3 convolution layers] --\
                                                             >-> [THINKING: 2 layers] -> ACTOR  -> 13 action chances
    self (48 numbers)   --> [BODY SENSE: 1 layer] ----------/                         -> CRITIC -> "how good is my situation?"

EYES: convolution layers slide small 3 x 3 "pattern detectors" over the view.
      Each detector learns to light up for one useful pattern, such as
      "water right next to me" or "something with glowing eyes nearby".
BODY SENSE: turns the 48 self numbers into features like "very thirsty".
THINKING: combines what it sees with how it feels.
ACTOR: a score for each of the 13 actions, turned into chances (softmax).
CRITIC: one number, its guess of how much future reward to expect.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical

from src.observe import NUM_CHANNELS, NUM_SELF, VIEW_SIZE
from src.agents import NUM_ACTIONS


class Brain(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.eyes = nn.Sequential(
            nn.Conv2d(NUM_CHANNELS, 32, kernel_size=3, padding=1), nn.ReLU(),            # 15 x 15
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),            # 8 x 8
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),            # 4 x 4
            nn.Flatten(),
        )
        with torch.no_grad():
            eye_size = self.eyes(torch.zeros(1, NUM_CHANNELS, VIEW_SIZE, VIEW_SIZE)).shape[1]
        self.body = nn.Sequential(nn.Linear(NUM_SELF, 64), nn.ReLU())
        self.thinking = nn.Sequential(
            nn.Linear(eye_size + 64, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.actor = nn.Linear(hidden, NUM_ACTIONS)
        self.critic = nn.Linear(hidden, 1)

        # Start the actor almost neutral (every action about equally likely),
        # so the newborn brain explores everything
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)

    def forward(self, view, self_features):
        features = torch.cat([self.eyes(view), self.body(self_features)], dim=1)
        thought = self.thinking(features)
        return self.actor(thought), self.critic(thought).squeeze(1)

    @torch.no_grad()
    def act(self, view, self_features, greedy=False):
        """Pick an action for each agent. Returns (actions, log_probs, values).
        greedy=True always picks the most likely action (used only for testing)."""
        logits, values = self(view, self_features)
        dist = Categorical(logits=logits)
        actions = logits.argmax(dim=1) if greedy else dist.sample()
        return actions, dist.log_prob(actions), values

    def evaluate(self, view, self_features, actions):
        """Used while learning: how likely the brain NOW finds the actions that
        were taken, how unsure it is overall (entropy), and its value guesses."""
        logits, values = self(view, self_features)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), values


def count_weights(model):
    return sum(p.numel() for p in model.parameters())


def save_brain(brain, path, extra=None):
    torch.save({"weights": brain.state_dict(), "extra": extra or {}}, path)


def load_brain(path, device="cpu"):
    data = torch.load(path, map_location=device, weights_only=False)
    brain = Brain().to(device)
    brain.load_state_dict(data["weights"])
    return brain, data.get("extra", {})
