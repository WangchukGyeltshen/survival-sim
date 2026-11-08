"""The HISTORY run: one world, never reset, recorded from start to finish.

Unlike training (many short worlds), this drops agents into ONE fresh world
and lets their story play out: generations are born, live and die, and
everything is written down so it can be studied (analyze.py) and rewatched
(replay.py).

    python experiment.py --brain runs/run2/latest.pt              agents keep learning as they live
    python experiment.py --brain runs/run2/latest.pt --frozen     brain never changes (instincts only)
    python experiment.py --blank                                  control: a brand-new, untrained brain
    python experiment.py ... --name exp1 --seed 7 --days 400 --no-watch

The run ends when everyone has died, after --days days (if given), or when
you close the window / press Ctrl+C. Results go to experiments/<name>/.
"""

import argparse
import os
import time

import numpy as np
import torch

import train
from src.simulation import Simulation
from src.observe import observe
from src.recorder import Recorder
from src.agents import NUM_ACTIONS
from src.brain import TribeBrains, load_into, saved_weights
from src.temperament import temperament


class HistoryRun(train.Trainer):
    """A Trainer that never resets the world and records everything."""

    def __init__(self, args):
        train_args = argparse.Namespace(name=os.path.join("..", "experiments", args.name, "brain"),
                                        resume=False, no_watch=args.no_watch, cpu=args.cpu,
                                        worlds=1)          # a history follows just one world
        super().__init__(train_args)
        self.args = args
        self.folder_out = os.path.join("experiments", args.name)
        if args.brain and not args.blank:
            data = torch.load(args.brain, map_location=self.device, weights_only=False)
            count = len(saved_weights(data))       # one brain per tribe, or one shared brain
            self.brain = TribeBrains(count).to(self.device)
            self.optimizers = [torch.optim.Adam(b.parameters(), lr=train.LEARNING_RATE, eps=1e-5)
                               for b in self.brain.brains]
            load_into(self.brain, data, args.brain)
            self.use_temperament = data.get("extra", {}).get("temperament", True)
            print(f"Loaded brain: {args.brain} ({count} brain{'s' if count > 1 else ''})")
        elif not args.blank:
            raise SystemExit("Give a brain with --brain runs/run2/latest.pt, or use --blank for an untrained one")
        else:
            print("Using a blank (untrained) brain: this is the control experiment")

        if args.no_temperament:
            self.use_temperament = False

        # One fixed world for the whole history
        self.set_world(Simulation(size=128, agents=200, seed=args.seed))
        if self.viewer:
            self.viewer.training = True
        info = {"name": args.name, "seed": args.seed, "brain": None if args.blank else args.brain,
                "learning": not (args.frozen or args.blank), "brains": self.brain.count,
                "temperament": self.use_temperament, "started": time.strftime("%Y-%m-%d %H:%M")}
        self.recorder = Recorder(self.sim, self.folder_out, info)
        self.tick_hooks.append(self.recorder.record_tick)

    def should_reset(self, k=0):
        return False                       # history never restarts

    def next_world(self, k=0):
        self.stop = True                   # everyone has died: the story is over

    def finished(self):
        if self.sim.agents.population == 0:
            print(f"Everyone has died on day {self.sim.world.day}.")
            return True
        if self.args.days and self.sim.world.day >= self.args.days:
            print(f"Reached day {self.args.days}.")
            return True
        return False

    def info_lines(self, stats=None):
        a = self.sim.agents
        lines = [f"HISTORY RUN  {self.args.name}",
                 f"day {self.sim.world.day}   alive {a.population}   born {a.births}",
                 "brain: " + ("frozen" if self.args.frozen else "learning as it lives")]
        if stats:
            lines.append(f"reward per agent per day: {stats['reward_per_day']:.2f}")
        return lines

    def run_frozen(self):
        """No learning: every tick, ask the brain, step the world, record."""
        sim = self.sim
        while not self.stop and not self.finished():
            a = sim.agents
            idx = np.flatnonzero(a.alive)
            view, feats = observe(sim, idx)
            memory = self.banks[0].get(idx, a.spawn_tick[idx])
            temper = torch.from_numpy(temperament(a, idx)).to(self.device) if self.use_temperament else None
            acts, _, _, memory = self.brain.act(torch.from_numpy(view).to(self.device),
                                                torch.from_numpy(feats).to(self.device), memory, temper,
                                                a.tribe[idx])
            self.banks[0].set(idx, memory)
            actions = np.zeros(a.count, dtype=np.int64)
            actions[idx] = acts.cpu().numpy()
            alive_before = a.alive.copy()
            sim.step(actions)
            for hook in self.tick_hooks:
                hook(actions, alive_before)
            if self.viewer and sim.world.tick % 50 == 0:
                self.viewer.training_info = self.info_lines()
            self.watch()

    def run_learning(self):
        """Same learning rounds as training, but in this one world."""
        while not self.stop and not self.finished():
            start = time.time()
            steps, bootstrap, stats = self.collect()
            if not steps:
                break
            self.advantages(steps, bootstrap)
            learn_stats = self.learn(steps)
            self.update += 1
            stats["curiosity"] = learn_stats["entropy"] / np.log(NUM_ACTIONS)
            self.save()
            if self.viewer:
                self.viewer.training_info = self.info_lines(stats)
            print(f"day {self.sim.world.day:4d} | alive {self.sim.agents.population:4d} | "
                  f"born {self.sim.agents.births:4d} | reward/day {stats['reward_per_day']:5.2f} | "
                  f"{time.time() - start:4.1f}s")

    def go(self):
        try:
            if self.args.frozen or self.args.blank:
                self.run_frozen()
            else:
                self.run_learning()
        except KeyboardInterrupt:
            pass
        self.recorder.close()
        if not (self.args.frozen or self.args.blank):
            self.save()
        print(f"History saved to {self.folder_out}/")
        print(f"  charts:  python analyze.py {self.folder_out}")
        print(f"  replay:  python replay.py {self.folder_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one recorded history")
    parser.add_argument("--brain", help="brain to use, e.g. runs/run2/latest.pt")
    parser.add_argument("--blank", action="store_true", help="use an untrained brain (control)")
    parser.add_argument("--frozen", action="store_true", help="don't learn during the run")
    parser.add_argument("--name", default="exp1", help="results go to experiments/<name>/")
    parser.add_argument("--seed", type=int, default=7, help="which world to generate")
    parser.add_argument("--days", type=int, default=0, help="stop after this many days (0 = never)")
    parser.add_argument("--no-watch", action="store_true", help="no window (faster)")
    parser.add_argument("--cpu", action="store_true", help="don't use the GPU")
    parser.add_argument("--no-temperament", action="store_true",
                        help="behaviour genes don't nudge choices (default: as the brain was trained)")
    run_args = parser.parse_args()
    if os.path.exists(os.path.join("experiments", run_args.name, "daily.csv")):
        raise SystemExit(f"experiments/{run_args.name} already exists; pick another --name")
    HistoryRun(run_args).go()
