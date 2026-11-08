"""The agents' instincts, given to them as rewards.

This is the ONLY teaching the agents get. Nothing here mentions food, water,
hunting, farming or weapons; agents must discover that those things help them
stay alive and have children.

    Instinct    When                              Reward
    survive     every tick alive                  +0.01   (so +1 per day)
    survive     dying (except of old age)         -3
    breed       a child is born (both parents)    +5
    parenting   each tick, per own child under    +0.002  (+0.2 per day; added in
                1 year old that is alive                   Stage 2, small like comfort)
    comfort     fed, watered, warm, rested,       +0.002  (only 20% of the
                healthy, not sick, not in pain             survival reward)
"""

import numpy as np

from src.agents import OLD_AGE, ADULT_AGE, TICKS_PER_YEAR

ALIVE_REWARD = 0.01
DEATH_PENALTY = -3.0
BIRTH_REWARD = 5.0
COMFORT_REWARD = 0.002
COMFORT_LEVEL = 60        # every need must be above this to feel comfortable
PARENT_REWARD = 0.002     # per young child alive, per tick


def compute_rewards(agents, alive_before, died):
    """Reward for every agent slot this tick.

    alive_before: who was alive at the start of the tick (they are the ones acting)
    died:         who died this tick
    Returns (rewards, done) where done marks agents whose life just ended."""
    a = agents
    rewards = np.zeros(a.count, dtype=np.float32)

    survived = alive_before & ~died
    rewards[survived] += ALIVE_REWARD

    # Death is bad, but dying of old age can't be avoided, so it isn't punished
    punish = died & (a.cause_of_death != OLD_AGE)
    rewards[punish] += DEATH_PENALTY

    for mother, father in a.new_parents:
        rewards[mother] += BIRTH_REWARD
        if father >= 0 and a.alive[father]:
            rewards[father] += BIRTH_REWARD

    # Parenting: a little reward for every young child of yours who is alive
    young = np.flatnonzero(a.alive & (a.age < ADULT_AGE * TICKS_PER_YEAR))
    for parents in (a.mother[young], a.father[young]):
        parents = parents[parents >= 0]
        parents = parents[survived[parents]]
        np.add.at(rewards, parents, PARENT_REWARD)

    comfy = survived & (a.sick == 0) & (a.pain == 0)
    for need in (a.health, a.food, a.water, a.energy, a.warmth):
        comfy &= need > COMFORT_LEVEL
    rewards[comfy] += COMFORT_REWARD

    return rewards, died
