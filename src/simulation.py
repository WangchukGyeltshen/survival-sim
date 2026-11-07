"""Ties the world and the agents together and runs them one tick at a time."""

from collections import Counter

import numpy as np

from src.world import World
from src.agents import Agents, NUM_ACTIONS, DEATH_CAUSES
from src.animals import Animals
from src.disasters import Disasters

START_AGENTS = 200


class Simulation:
    def __init__(self, size=128, agents=START_AGENTS, seed=None):
        self.rng = np.random.default_rng(seed)
        self.world = World(size=size, seed=seed)
        self.agents = Agents(self.world, agents, self.rng)
        self.animals = Animals(self.world, self.rng)
        self.disasters = Disasters(self.world, self.rng)
        self.deaths = Counter()   # cause of death -> how many
        self.discoveries = []     # (tick, message) the first time each activity happens

    def choose_actions(self):
        """For now every agent picks a completely random action.
        In phase 10 this is replaced by the learning "brain"."""
        return self.rng.integers(0, NUM_ACTIONS, self.agents.count)

    def step(self):
        actions = self.choose_actions()
        self.disasters.step()
        self.agents.step(actions, self.disasters, self.animals)
        self.animals.step(self.agents)
        dead = self.agents.resolve_deaths()
        self.check_discoveries()
        for i in np.flatnonzero(dead):
            self.deaths[DEATH_CAUSES[self.agents.cause_of_death[i]]] += 1
        self.world.step()

    def check_discoveries(self):
        """Note the first time anyone does something new (a spear, a planted seed...)."""
        known = {m for _, m in self.discoveries}
        for activity in self.agents.counts:
            message = f"First time: {activity}"
            if message not in known:
                self.discoveries.append((self.world.tick, message))

    @property
    def tick(self):
        return self.world.tick
