"""Train the shared brain with PPO, and (optionally) watch it learn live.

    python train.py                  train a new brain, with the live window
    python train.py --resume         carry on from the last saved brain
    python train.py --no-watch       train without the window (fastest)
    python train.py --name run2      keep a separate experiment in runs/run2

In the window: T = turbo (stop drawing, train at full speed), SPACE = pause,
+/- = slow down / speed up (MAX = as fast as possible). Closing the window
saves the brain and stops training. Ctrl+C in the terminal does the same.

HOW ONE ROUND OF LEARNING WORKS
1. COLLECT: every living agent uses the current brain for a while (until about
   32,000 decisions have been made). For each decision we store what the agent
   saw, what it did, how sure the brain was, the critic's guess and the reward.
2. JUDGE: for every decision, work out its ADVANTAGE: did things turn out
   better or worse than the critic expected? (using the GAE method below)
3. LEARN: nudge the brain's weights so better-than-expected actions become a
   bit more likely and worse ones a bit less likely, and so the critic's
   guesses get more accurate. PPO's rule: never change too much at once.
4. Save the brain, write a line to the log, and start the next round.
"""

import argparse
import csv
import os
import time
from collections import Counter

import numpy as np
import torch

from src.simulation import Simulation
from src.observe import observe
from src.brain import Brain, save_brain
from src.rewards import compute_rewards
from src.agents import NUM_ACTIONS, OLD_AGE, DEATH_CAUSES
from src.world import TICKS_PER_DAY

# ---------------- settings ----------------
BATCH_SAMPLES = 32_768     # decisions collected before each learning round
MAX_ROLLOUT_TICKS = 512    # ...or this many ticks, whichever comes first
MINIBATCH = 4096           # decisions looked at together in one learning step
EPOCHS = 3                 # how many times each batch is studied
LEARNING_RATE = 2.5e-4     # size of each nudge to the weights
GAMMA = 0.995              # how much the future matters (0.995 = about 2 days ahead)
GAE_LAMBDA = 0.95          # smooths the advantage estimate
CLIP = 0.2                 # PPO safety rule: a decision's chance can change by at most ~20% per round
ENTROPY_BONUS = 0.01       # keeps agents curious (rewards being unsure)
VALUE_WEIGHT = 0.5         # how much the critic's accuracy matters in learning
MAX_GRAD = 0.5             # caps any single nudge
MIN_POPULATION = 20        # fewer alive than this: start a fresh world
MAX_WORLD_DAYS = 100       # also start a fresh world after this long (variety)
SNAPSHOT_EVERY = 50        # keep a copy of the brain every 50 rounds
DRAW_FPS = 30


def new_world():
    seed = int(np.random.default_rng().integers(1_000_000))
    return Simulation(size=128, agents=200, seed=seed)


class Trainer:
    def __init__(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
        self.folder = os.path.join("runs", args.name)
        os.makedirs(self.folder, exist_ok=True)
        self.brain = Brain().to(self.device)
        self.optimizer = torch.optim.Adam(self.brain.parameters(), lr=LEARNING_RATE, eps=1e-5)
        self.update = 0
        self.total_ticks = 0
        self.total_samples = 0
        self.worlds = 1
        latest = os.path.join(self.folder, "latest.pt")
        if args.resume and os.path.exists(latest):
            data = torch.load(latest, map_location=self.device, weights_only=False)
            self.brain.load_state_dict(data["weights"])
            self.optimizer.load_state_dict(data["optimizer"])
            self.update = data["extra"]["update"]
            self.total_ticks = data["extra"]["total_ticks"]
            self.total_samples = data["extra"].get("total_samples", 0)
            self.worlds = data["extra"].get("worlds", 1)
            print(f"Resumed from round {self.update}")
        self.sim = new_world()
        self.viewer = None
        if not args.no_watch:
            from src.viewer import Viewer, SPEEDS
            self.viewer = Viewer(self.sim)
            self.viewer.training = True
            self.viewer.speed_index = len(SPEEDS) - 1   # MAX
            self.speeds = SPEEDS
        self.last_draw = 0.0
        self.stop = False
        self.log_path = os.path.join(self.folder, "log.csv")
        self.recent_life = []        # days survived by agents who died recently
        self.tick_hooks = []         # functions(actions, acted) called after every tick
        print(f"Training on: {self.device}  |  saving to: {self.folder}")

    # ---------------- the window ----------------
    def watch(self):
        """Keep the window alive. Called after every tick."""
        v = self.viewer
        if v is None:
            return
        v.after_tick()
        speed = self.speeds[v.speed_index]
        limited = v.speed_index < len(self.speeds) - 1
        now = time.time()
        if limited:
            # Slow motion: wait so the world runs at the chosen ticks per second
            time.sleep(max(0.0, 1 / speed - (now - self.last_draw)))
        elif now - self.last_draw < (0.5 if v.turbo else 1 / DRAW_FPS):
            return
        self.draw()
        while v.paused and not self.stop:
            self.draw()
            time.sleep(1 / DRAW_FPS)

    def draw(self):
        v = self.viewer
        if not v.handle_events():
            self.stop = True
        now = time.time()
        v.frame(min(now - self.last_draw, 0.1))
        self.last_draw = now

    def info_lines(self, stats=None):
        life = np.mean(self.recent_life[-300:]) if self.recent_life else 0
        lines = [f"TRAINING  round {self.update}",
                 f"world #{self.worlds}   ticks trained {self.total_ticks:,}",
                 f"avg days survived (recent): {life:.2f}"]
        if stats:
            lines += [f"reward per agent per day: {stats['reward_per_day']:.2f}",
                      f"curiosity (entropy): {stats['curiosity']:.0%}",
                      f"births this round: {stats['births']}"]
        return lines

    # ---------------- 1. COLLECT ----------------
    def collect(self):
        sim, brain = self.sim, self.brain
        steps = []          # one entry per tick
        samples = 0
        births = 0
        rewards_total = 0.0
        agent_ticks = 0
        deaths = Counter()
        while samples < BATCH_SAMPLES and len(steps) < MAX_ROLLOUT_TICKS and not self.stop:
            a = sim.agents
            idx = np.flatnonzero(a.alive)
            if len(idx) == 0:
                break
            view, feats = observe(sim, idx)
            view_t = torch.from_numpy(view).to(self.device)
            feats_t = torch.from_numpy(feats).to(self.device)
            acts, logp, values = brain.act(view_t, feats_t)
            actions = np.zeros(a.count, dtype=np.int64)
            actions[idx] = acts.cpu().numpy()

            alive_before = a.alive.copy()
            sim.step(actions)
            for hook in self.tick_hooks:
                hook(actions, alive_before)
            died = alive_before & ~a.alive
            rewards, done = compute_rewards(a, alive_before, died)

            steps.append({
                "idx": idx,
                "view": (view * 255).astype(np.uint8),     # stored small to save memory
                "feats": feats,
                "actions": actions[idx],
                "logp": logp.cpu().numpy(),
                "values": values.cpu().numpy(),
                "rewards": rewards[idx],
                "done": done[idx],
            })
            samples += len(idx)
            agent_ticks += len(idx)
            rewards_total += float(rewards[idx].sum())
            births += len(a.new_parents)
            for i in np.flatnonzero(died):
                deaths[DEATH_CAUSES[a.cause_of_death[i]]] += 1
                if a.cause_of_death[i] != OLD_AGE:
                    self.recent_life.append((a.death_tick[i] - a.spawn_tick[i]) / TICKS_PER_DAY)
            self.watch()

        # The critic's guess for agents still alive at the end ("what comes next")
        bootstrap = np.zeros(sim.agents.count, dtype=np.float32)
        idx = np.flatnonzero(sim.agents.alive)
        if len(idx):
            view, feats = observe(sim, idx)
            with torch.no_grad():
                _, v = brain(torch.from_numpy(view).to(self.device), torch.from_numpy(feats).to(self.device))
            bootstrap[idx] = v.cpu().numpy()
        self.total_ticks += len(steps)
        self.total_samples += samples
        stats = {"samples": samples, "ticks": len(steps), "births": births, "deaths": deaths,
                 "reward_per_day": rewards_total / max(agent_ticks, 1) * TICKS_PER_DAY}
        return steps, bootstrap, stats

    # ---------------- 2. JUDGE (advantages) ----------------
    def advantages(self, steps, bootstrap):
        """GAE: work backwards through time. For each decision:
            surprise = reward + GAMMA * (critic's guess for the next tick) - (critic's guess now)
            advantage = surprise + GAMMA * LAMBDA * (next tick's advantage)
        A death ends an agent's story, so nothing after it counts."""
        n = len(bootstrap)
        next_value = bootstrap.copy()
        next_adv = np.zeros(n, dtype=np.float32)
        for step in reversed(steps):
            idx, done = step["idx"], step["done"]
            nv = np.where(done, 0.0, next_value[idx])
            na = np.where(done, 0.0, next_adv[idx])
            surprise = step["rewards"] + GAMMA * nv - step["values"]
            adv = surprise + GAMMA * GAE_LAMBDA * na
            step["adv"] = adv.astype(np.float32)
            step["returns"] = (adv + step["values"]).astype(np.float32)
            next_value[idx] = step["values"]
            next_adv[idx] = adv

    # ---------------- 3. LEARN (PPO) ----------------
    def learn(self, steps):
        cat = lambda key: np.concatenate([s[key] for s in steps])
        view, feats = cat("view"), cat("feats")
        actions, old_logp = cat("actions"), cat("logp")
        adv, returns = cat("adv"), cat("returns")
        # Scale advantages to an average of 0 and spread of 1 (makes learning steadier)
        adv = ((adv - adv.mean()) / (adv.std() + 1e-8)).astype(np.float32)
        n = len(actions)
        totals = Counter()
        updates = 0
        for _ in range(EPOCHS):
            order = np.random.permutation(n)
            for start in range(0, n, MINIBATCH):
                mb = order[start:start + MINIBATCH]
                if len(mb) < MINIBATCH // 4:
                    continue   # skip a tiny leftover group
                v_t = torch.from_numpy(view[mb]).to(self.device).float() / 255
                f_t = torch.from_numpy(feats[mb]).to(self.device)
                a_t = torch.from_numpy(actions[mb]).to(self.device)
                old_t = torch.from_numpy(old_logp[mb]).to(self.device)
                adv_t = torch.from_numpy(adv[mb]).to(self.device)
                ret_t = torch.from_numpy(returns[mb]).to(self.device)

                new_logp, entropy, value = self.brain.evaluate(v_t, f_t, a_t)
                # How much more (or less) likely is this action now than when it was taken?
                ratio = torch.exp(new_logp - old_t)
                # PPO: follow the advantage, but don't let the ratio move past 1 +/- CLIP
                policy_loss = -torch.min(ratio * adv_t,
                                         torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * adv_t).mean()
                value_loss = ((value - ret_t) ** 2).mean()
                entropy_mean = entropy.mean()
                loss = policy_loss + VALUE_WEIGHT * value_loss - ENTROPY_BONUS * entropy_mean

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.brain.parameters(), MAX_GRAD)
                self.optimizer.step()

                totals["policy_loss"] += policy_loss.item()
                totals["value_loss"] += value_loss.item()
                totals["entropy"] += entropy_mean.item()
                updates += 1
            if self.viewer:
                self.draw()   # keep the window responsive while learning
        return {k: v / max(updates, 1) for k, v in totals.items()}

    # ---------------- 4. SAVE and LOG ----------------
    def save(self):
        extra = {"update": self.update, "total_ticks": self.total_ticks,
                 "total_samples": self.total_samples, "worlds": self.worlds}
        data = {"weights": self.brain.state_dict(), "optimizer": self.optimizer.state_dict(), "extra": extra}
        torch.save(data, os.path.join(self.folder, "latest.pt"))
        if self.update % SNAPSHOT_EVERY == 0:
            save_brain(self.brain, os.path.join(self.folder, f"brain_round{self.update:05d}.pt"), extra)

    def log(self, stats, learn_stats, seconds):
        life = np.mean(self.recent_life[-300:]) if self.recent_life else 0
        row = {"round": self.update, "world": self.worlds, "world_day": self.sim.world.day,
               "ticks": self.total_ticks, "samples": self.total_samples,
               "population": self.sim.agents.population, "births": stats["births"],
               "avg_days_survived": round(life, 3),
               "reward_per_day": round(stats["reward_per_day"], 4),
               "curiosity": round(stats["curiosity"], 4),
               "policy_loss": round(learn_stats["policy_loss"], 5),
               "value_loss": round(learn_stats["value_loss"], 5),
               "seconds": round(seconds, 1)}
        for cause in DEATH_CAUSES[1:]:
            row[f"died_{cause}"] = stats["deaths"].get(cause, 0)
        new_file = not os.path.exists(self.log_path)
        with open(self.log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if new_file:
                writer.writeheader()
            writer.writerow(row)
        top = ", ".join(f"{c} {n}" for c, n in stats["deaths"].most_common(3))
        print(f"round {self.update:5d} | world {self.worlds} day {self.sim.world.day:3d} | "
              f"pop {self.sim.agents.population:4d} | survived {life:5.2f} days | "
              f"reward/day {stats['reward_per_day']:5.2f} | curiosity {stats['curiosity']:4.0%} | "
              f"births {stats['births']:3d} | deaths: {top} | {seconds:4.1f}s")

    # ---------------- main loop ----------------
    def run(self):
        try:
            while not self.stop:
                start = time.time()
                steps, bootstrap, stats = self.collect()
                if self.stop or not steps:
                    if not steps:
                        self.next_world()
                    continue
                self.advantages(steps, bootstrap)
                learn_stats = self.learn(steps)
                self.update += 1
                stats["curiosity"] = learn_stats["entropy"] / np.log(NUM_ACTIONS)
                self.save()
                self.log(stats, learn_stats, time.time() - start)
                if self.viewer:
                    self.viewer.training_info = self.info_lines(stats)
                if self.should_reset():
                    self.next_world()
        except KeyboardInterrupt:
            pass
        self.save()
        print(f"Stopped. Brain saved to {self.folder}/latest.pt")

    def should_reset(self):
        return (self.sim.agents.population < MIN_POPULATION
                or self.sim.world.day >= MAX_WORLD_DAYS)

    def next_world(self):
        self.sim = new_world()
        self.worlds += 1
        if self.viewer:
            self.viewer.set_sim(self.sim)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the survival brain")
    parser.add_argument("--name", default="run1", help="experiment name (folder in runs/)")
    parser.add_argument("--resume", action="store_true", help="continue from runs/<name>/latest.pt")
    parser.add_argument("--no-watch", action="store_true", help="train without the window")
    parser.add_argument("--cpu", action="store_true", help="don't use the GPU")
    Trainer(parser.parse_args()).run()
