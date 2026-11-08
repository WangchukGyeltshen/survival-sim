"""Rewatch a recorded history run (made with experiment.py).

    python replay.py experiments/exp1

Extra keys in the replay window:
    PAGE DOWN / PAGE UP    jump 10 days forward / back
    HOME                   back to the start
    (plus all the normal keys: SPACE pause, +/- speed, click agents, log...)

How it works: the recording has a full copy of the world every 10 days and
every action every agent took. Replaying loads the nearest copy and re-runs
the world with exactly the same actions, so everything happens again exactly
as it did the first time.
"""

import argparse
import glob
import json
import os

import numpy as np
import pygame

from src.recorder import load_snapshot, SNAPSHOT_DAYS
from src.viewer import Viewer, SPEEDS
from src.world import TICKS_PER_DAY


class Replay:
    def __init__(self, folder):
        self.folder = folder
        replay = os.path.join(folder, "replay")
        self.snapshots = sorted(int(os.path.basename(p)[4:9]) for p in glob.glob(os.path.join(replay, "day_*.pkl")))
        self.chunks = {int(os.path.basename(p)[8:13]): p for p in glob.glob(os.path.join(replay, "actions_*.npz"))}
        if not self.snapshots:
            raise SystemExit(f"No replay found in {folder}")
        self.last_day = max(day + len(np.load(p)["actions"]) / TICKS_PER_DAY for day, p in self.chunks.items()) \
            if self.chunks else 0
        self.loaded_chunk = (None, None)
        with open(os.path.join(folder, "info.json")) as f:
            self.info = json.load(f)
        self.viewer = None
        self.ended = False

    def actions_for(self, sim):
        """The recorded actions for the tick about to happen."""
        start_day = (sim.world.day // SNAPSHOT_DAYS) * SNAPSHOT_DAYS
        if self.loaded_chunk[0] != start_day:
            path = self.chunks.get(start_day)
            self.loaded_chunk = (start_day, np.load(path)["actions"] if path else None)
        actions = self.loaded_chunk[1]
        k = sim.world.tick - start_day * TICKS_PER_DAY
        if actions is None or k >= len(actions):
            self.ended = True
            self.viewer.paused = True
            return np.zeros(sim.agents.count, dtype=np.int64)
        return actions[k].astype(np.int64)

    def jump_to(self, day):
        """Load the nearest saved copy at or before `day`, then fast-forward."""
        day = max(0, min(day, int(self.last_day)))
        base = max(d for d in self.snapshots if d <= day)
        sim = load_snapshot(os.path.join(self.folder, "replay", f"day_{base:05d}.pkl"))
        sim.policy = self.actions_for
        self.ended = False
        while sim.world.day < day and not self.ended:
            sim.step()
        self.viewer.set_sim(sim)
        self.viewer.paused = self.ended

    def run(self):
        first = load_snapshot(os.path.join(self.folder, "replay", "day_00000.pkl"))
        first.policy = self.actions_for
        self.viewer = v = Viewer(first)
        v.speed_index = 3
        v.key_handlers = {
            pygame.K_PAGEDOWN: lambda: self.jump_to(v.world.day + 10),
            pygame.K_PAGEUP: lambda: self.jump_to(v.world.day - 10),
            pygame.K_HOME: lambda: self.jump_to(0),
        }
        brain = self.info.get("brain") or "blank brain"
        running, tick_timer = True, 0.0
        while running:
            running = v.handle_events()
            dt = v.clock.tick(60) / 1000
            if not v.paused:
                tick_timer += dt * SPEEDS[v.speed_index]
                while tick_timer >= 1 and not self.ended:
                    v.sim.step()
                    v.after_tick()
                    tick_timer -= 1
            v.training_info = [f"REPLAY  {self.info['name']}",
                               f"day {v.world.day} of {int(self.last_day)}" + ("   (end)" if self.ended else ""),
                               f"brain: {brain}" + ("" if self.info.get("learning") else " (frozen)"),
                               "PAGE UP / DOWN: jump 10 days, HOME: start"]
            v.frame(dt)
        pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rewatch a recorded history run")
    parser.add_argument("folder", help="e.g. experiments/exp1")
    Replay(parser.parse_args().folder).run()
