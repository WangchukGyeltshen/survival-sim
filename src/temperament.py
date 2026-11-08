"""Temperament: how behaviour genes nudge the brain's choices.

The brain gives every action a score; the scores become chances (softmax).
Before that happens, each agent's behaviour genes adjust the scores:

    aggression  +  attack a person (and a little: attack an animal)
    generosity  +  give
    drive       +  mate
    industry    +  craft, build, plant
    curiosity   makes all choices more random (higher) or more set in their
                ways (lower), by dividing the scores ("temperature")

A gene of +1 makes its actions about 2.7 times as likely as the brain alone
would choose (e to the power 1); -1 makes them about 2.7 times less likely.
The brain can still overrule a gene by learning stronger scores, so genes are
leanings, not commands. Because they are inherited (with mutation), evolution
can favour whatever temperament survives and breeds best, while the shared
brain learns how to make the most of each temperament. The brain also senses
its own genes, so it knows what kind of person it is.
"""

import numpy as np
import torch

from src.agents import (BEHAVIOUR_GENES, NUM_ACTIONS, ATTACK_PERSON, ATTACK_ANIMAL, GIVE, MATE,
                        CRAFT, BUILD, PLANT)

NUDGES = {
    "aggression": {ATTACK_PERSON: 1.0, ATTACK_ANIMAL: 0.5},
    "generosity": {GIVE: 1.0},
    "drive": {MATE: 1.0},
    "industry": {CRAFT: 0.6, BUILD: 0.6, PLANT: 0.6},
}
CURIOSITY_STRENGTH = 0.5      # curiosity +1: scores divided by 1.65; -1: multiplied by 1.65

# (genes x actions) table of nudges, in BEHAVIOUR_GENES order (curiosity row stays 0)
NUDGE_TABLE = np.zeros((len(BEHAVIOUR_GENES), NUM_ACTIONS), dtype=np.float32)
for row, gene in enumerate(BEHAVIOUR_GENES):
    for action, amount in NUDGES.get(gene, {}).items():
        NUDGE_TABLE[row, action] = amount
CURIOSITY = BEHAVIOUR_GENES.index("curiosity")


def temperament(agents, idx):
    """The behaviour genes of the given agents: float32 array (N, genes)."""
    return np.stack([agents.genes[g][idx] for g in BEHAVIOUR_GENES], axis=1).astype(np.float32)


def shape_scores(logits, temper):
    """Apply each agent's temperament to the brain's action scores (torch)."""
    if temper is None:
        return logits
    table = torch.as_tensor(NUDGE_TABLE, device=logits.device)
    temperature = torch.exp(CURIOSITY_STRENGTH * temper[:, CURIOSITY])
    return logits / temperature[:, None] + temper @ table
