"""Several worlds at once, for faster and more varied training (Stage 4).

World 0 runs in the main process, so the live window can show it. The other
worlds each run in their own process (a "worker"), on another CPU core.
Every tick:

    1. every world sends what its agents sense
    2. the main process asks the brains for everyone's actions (one go on the GPU)
    3. each world takes its actions, steps forward, and reports back the
       rewards, deaths, births, and what its agents sense next

Workers step their worlds at the same time as world 0, so 4 worlds take
little longer per tick than 1. Learning also gets a better mix: each round
holds decisions from several worlds, in different seasons and situations.
"""

import multiprocessing as mp

# "spawn" starts each worker as a fresh Python (the only way on Windows, and
# safe with the GPU everywhere)
CTX = mp.get_context("spawn")

import numpy as np

from src.simulation import Simulation
from src.observe import observe
from src.rewards import compute_rewards
from src.temperament import temperament
from src.agents import OLD_AGE, DEATH_CAUSES
from src.world import TICKS_PER_DAY


def make_sim(seed=None):
    if seed is None:
        seed = int(np.random.default_rng().integers(1_000_000))
    return Simulation(size=128, agents=200, seed=seed)


class World:
    """One world, with what training needs from it."""

    def __init__(self, sim):
        self.sim = sim
        self.pending = None

    def observe(self):
        """What every living agent senses, plus a little about the world."""
        sim = self.sim
        a = sim.agents
        idx = np.flatnonzero(a.alive)
        obs = {"idx": idx, "day": sim.world.day, "population": len(idx), "count": a.count}
        if len(idx):
            view, feats = observe(sim, idx)
            obs.update(view=(view * 255).astype(np.uint8),     # small to send and to store
                       feats=feats,
                       tribe=a.tribe[idx].copy(),
                       spawn=a.spawn_tick[idx].astype(np.int64),
                       temper=temperament(a, idx))
        return obs

    def step(self, idx, chosen):
        """Take the chosen actions (one per agent in idx) and move time forward.
        Returns (result, full actions array, who was alive before)."""
        sim = self.sim
        a = sim.agents
        actions = np.zeros(a.count, dtype=np.int64)
        actions[idx] = chosen
        alive_before = a.alive.copy()
        sim.step(actions)
        died = alive_before & ~a.alive
        rewards, done = compute_rewards(a, alive_before, died)
        deaths = [(DEATH_CAUSES[a.cause_of_death[i]],
                   (a.death_tick[i] - a.spawn_tick[i]) / TICKS_PER_DAY,
                   bool(a.cause_of_death[i] == OLD_AGE))
                  for i in np.flatnonzero(died)]
        result = {"rewards": rewards[idx], "done": done[idx], "deaths": deaths,
                  "births": len(a.new_parents)}
        return result, actions, alive_before

    # The same "start, then finish" steps as a worker, so the trainer can treat
    # every world alike. World 0 does its work in finish(), while the workers run.
    def start_step(self, idx, chosen):
        self.pending = (idx, chosen)

    def finish_step(self):
        result, actions, alive_before = self.step(*self.pending)
        result["obs"] = self.observe()
        result["actions"], result["alive_before"] = actions, alive_before
        return result

    def reset(self, seed=None):
        self.sim = make_sim(seed)
        return self.observe()

    def close(self):
        pass


def _worker(conn, seed):
    """Runs in its own process: owns one world and does what it is told."""
    try:
        world = World(make_sim(seed))
        while True:
            command, arg = conn.recv()
            if command == "step":
                result, _, _ = world.step(*arg)
                result["obs"] = world.observe()
                conn.send(result)
            elif command == "observe":
                conn.send(world.observe())
            elif command == "reset":
                conn.send(world.reset(arg))
            elif command == "close":
                break
    except (KeyboardInterrupt, EOFError, BrokenPipeError):
        pass                                # training was stopped: just finish quietly


class RemoteWorld:
    """A world running in a worker process."""

    def __init__(self, seed=None):
        parent, child = CTX.Pipe()
        self.conn = parent
        self.process = CTX.Process(target=_worker, args=(child, seed), daemon=True)
        self.process.start()
        self.sim = None                     # lives in the other process

    def observe(self):
        self.conn.send(("observe", None))
        return self.conn.recv()

    def start_step(self, idx, chosen):
        self.conn.send(("step", (idx, chosen)))

    def finish_step(self):
        return self.conn.recv()

    def reset(self, seed=None):
        self.conn.send(("reset", seed))
        return self.conn.recv()

    def close(self):
        try:
            self.conn.send(("close", None))
        except (BrokenPipeError, OSError):
            pass
        self.process.join(timeout=2)
        if self.process.is_alive():
            self.process.terminate()
