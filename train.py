"""Train the tribe brains with PPO, and (optionally) watch them learn live.

    python train.py                  train new brains, with the live window
    python train.py --resume         carry on from the last saved brains
    python train.py --no-watch       train without the window (fastest)
    python train.py --name run2      keep a separate experiment in runs/run2
    python train.py --brains 1       a new run with ONE brain shared by all tribes
    python train.py --worlds 6       run 6 worlds at once (default 4; 1 = old behaviour)

Each tribe has its own brain (Stage 3), trained only on its own members'
experience. Resuming a run saved with one shared brain splits it: every
tribe's brain starts as a copy of it, and from then on they grow apart.

Several worlds run at once (Stage 4, src/worlds.py): world 0 in this process
(the one the window shows), the others in worker processes on other CPU
cores. Each round of learning mixes decisions from all of them.

In the window: T = turbo (stop drawing, train at full speed), SPACE = pause,
+/- = slow down / speed up (MAX = as fast as possible). Closing the window
saves the brain and stops training. Ctrl+C in the terminal does the same.

HOW ONE ROUND OF LEARNING WORKS
1. COLLECT: every living agent, in every world, uses the current brains for a
   while (until about 32,000 decisions have been made). For each decision we store what the agent
   saw, what it did, how sure the brain was, the critic's guess and the reward.
2. JUDGE: for every decision, work out its ADVANTAGE: did things turn out
   better or worse than the critic expected? (using the GAE method below)
3. LEARN: nudge the brain's weights so better-than-expected actions become a
   bit more likely and worse ones a bit less likely, and so the critic's
   guesses get more accurate. PPO's rule: never change too much at once.
   Because agents have MEMORY, decisions are studied in short stretches of one
   agent's life (SEQ_LEN ticks in a row), so the brain can learn what was
   worth remembering a few ticks earlier. Each tribe's brain learns only from
   its own tribe's decisions, with its own optimizer.
4. Save the brain, write a line to the log, and start the next round.
"""

import argparse
import csv
import os
import time
from collections import Counter

import numpy as np
import torch

from src.worlds import World, RemoteWorld, make_sim
from src.brain import TribeBrains, MemoryBank, save_brains, load_into
from src.agents import NUM_ACTIONS, DEATH_CAUSES, NUM_TRIBES
from src.world import TICKS_PER_DAY

# ---------------- settings ----------------
BATCH_SAMPLES = 32_768     # decisions collected before each learning round
MAX_ROLLOUT_TICKS = 512    # ...or this many ticks, whichever comes first
MINIBATCH = 4096           # decisions looked at together in one learning step
SEQ_LEN = 16               # memory is learned over stretches of this many ticks of one life
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
WORLDS = 4                 # worlds running at once (each extra one uses another CPU core)
DRAW_FPS = 30


def new_world():
    return make_sim()


class Trainer:
    def __init__(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
        self.folder = os.path.join("runs", args.name)
        os.makedirs(self.folder, exist_ok=True)
        latest = os.path.join(self.folder, "latest.pt")
        data = None
        count = getattr(args, "brains", None) or NUM_TRIBES
        if args.resume and os.path.exists(latest):
            data = torch.load(latest, map_location=self.device, weights_only=False)
            if "brains" in data:
                count = len(data["brains"])          # keep however many brains the run has
        self.brain = TribeBrains(count).to(self.device)
        self.optimizers = [torch.optim.Adam(b.parameters(), lr=LEARNING_RATE, eps=1e-5)
                           for b in self.brain.brains]
        self.parent = None
        self.surgeries = []
        self.update = 0
        self.total_ticks = 0
        self.total_samples = 0
        self.worlds = 1
        if data is not None:
            split = load_into(self.brain, data, latest)
            if "optimizers" in data:
                for opt, state in zip(self.optimizers, data["optimizers"]):
                    opt.load_state_dict(state)
            elif "optimizer" in data:          # one shared brain's optimizer: every copy gets it
                for opt in self.optimizers:
                    opt.load_state_dict(data["optimizer"])
            if split:
                print(f"Split the shared brain into {count} tribe brains (each starts as a copy)")
            extra = data.get("extra", {})
            self.update = extra.get("update", 0)
            self.total_ticks = extra.get("total_ticks", 0)
            self.total_samples = extra.get("total_samples", 0)
            self.worlds = extra.get("worlds", 1)
            self.parent = extra.get("parent")
            self.surgeries = extra.get("surgeries", [])
            print(f"Resumed from round {self.update}"
                  + (f" (brain grown from {self.parent})" if self.parent else ""))
        # World 0 here (shown in the window), the rest in worker processes
        count_worlds = max(1, getattr(args, "worlds", None) or 1)
        self.envs = [World(new_world())] + [RemoteWorld() for _ in range(count_worlds - 1)]
        self.banks = [None] * count_worlds        # each world's agents' memories
        self.obs = [None] * count_worlds          # what each world's agents sense right now
        self.viewer = None
        self.set_world(self.envs[0].sim)
        for k in range(1, count_worlds):
            self.obs[k] = self.envs[k].observe()
            self.banks[k] = MemoryBank(self.brain, self.obs[k]["count"], self.device)
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
        print(f"Training on: {self.device}  |  worlds at once: {count_worlds}  |  saving to: {self.folder}")

    @property
    def sim(self):
        """World 0 (the one in this process, shown in the window)."""
        return self.envs[0].sim

    def population(self):
        return sum(o["population"] for o in self.obs)

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
                 f"worlds at once: {len(self.envs)} (showing world 1)   alive in all: {self.population()}",
                 f"avg days survived (recent): {life:.2f}",
                 f"brains: {self.brain.count}" + (" (one per tribe)" if self.brain.count > 1 else " (shared)")]
        if stats:
            lines += [f"reward per agent per day: {stats['reward_per_day']:.2f}",
                      f"curiosity (entropy): {stats['curiosity']:.0%}",
                      f"births this round: {stats['births']}"]
        return lines

    # ---------------- 1. COLLECT ----------------
    def collect(self):
        brain = self.brain
        steps = []          # one entry per world per tick
        samples = 0
        births = 0
        rewards_total = 0.0
        ticks = 0
        deaths = Counter()
        while samples < BATCH_SAMPLES and ticks < MAX_ROLLOUT_TICKS and not self.stop:
            live = [k for k, o in enumerate(self.obs) if o["population"]]
            if not live:
                break
            # Everyone in every world decides at once (one go per tribe brain)
            cat = lambda key: np.concatenate([self.obs[k][key] for k in live])
            view, feats, tribes, temper = cat("view"), cat("feats"), cat("tribe"), cat("temper")
            memory = torch.cat([self.banks[k].get(self.obs[k]["idx"], self.obs[k]["spawn"]) for k in live])
            view_t = torch.from_numpy(view).to(self.device).float() / 255
            feats_t = torch.from_numpy(feats).to(self.device)
            acts, logp, values, new_memory = brain.act(view_t, feats_t, memory,
                                                       torch.from_numpy(temper).to(self.device), tribes)
            acts, logp, values = acts.cpu().numpy(), logp.cpu().numpy(), values.cpu().numpy()
            memory_np = memory.cpu().numpy()
            which = brain.which(tribes)

            # Hand each world its actions: workers start stepping right away,
            # world 0 steps in this process while they work
            parts = np.cumsum([0] + [self.obs[k]["population"] for k in live])
            for j, k in enumerate(live):
                rows = slice(parts[j], parts[j + 1])
                self.banks[k].set(self.obs[k]["idx"], new_memory[rows])
                self.envs[k].start_step(self.obs[k]["idx"], acts[rows])
            for j, k in enumerate(live):
                rows = slice(parts[j], parts[j + 1])
                o = self.obs[k]
                result = self.envs[k].finish_step()
                if k == 0:
                    for hook in self.tick_hooks:
                        hook(result["actions"], result["alive_before"])
                steps.append({
                    "world": k,
                    "idx": o["idx"],
                    "spawn": o["spawn"],                      # tells apart people who shared a slot
                    "memory": memory_np[rows],                # what they remembered before deciding
                    "temper": o["temper"],
                    "brain": which[rows],
                    "view": o["view"],
                    "feats": o["feats"],
                    "actions": acts[rows],
                    "logp": logp[rows],
                    "values": values[rows],
                    "rewards": result["rewards"],
                    "done": result["done"],
                })
                samples += o["population"]
                rewards_total += float(result["rewards"].sum())
                births += result["births"]
                for cause, days, old_age in result["deaths"]:
                    deaths[cause] += 1
                    if not old_age:
                        self.recent_life.append(days)
                self.obs[k] = result["obs"]
            ticks += 1
            self.watch()

        # The critic's guess for agents still alive at the end ("what comes next")
        bootstrap = {}
        for k, o in enumerate(self.obs):
            bootstrap[k] = np.zeros(o["count"], dtype=np.float32)
            if o["population"]:
                with torch.no_grad():
                    v = brain.value(torch.from_numpy(o["view"]).to(self.device).float() / 255,
                                    torch.from_numpy(o["feats"]).to(self.device),
                                    self.banks[k].get(o["idx"], o["spawn"]), o["tribe"])
                bootstrap[k][o["idx"]] = v.cpu().numpy()
        self.total_ticks += ticks * len(self.envs)
        self.total_samples += samples
        stats = {"samples": samples, "ticks": ticks, "births": births, "deaths": deaths,
                 "reward_per_day": rewards_total / max(samples, 1) * TICKS_PER_DAY}
        return steps, bootstrap, stats

    # ---------------- 2. JUDGE (advantages) ----------------
    def advantages(self, steps, bootstrap):
        """GAE: work backwards through time. For each decision:
            surprise = reward + GAMMA * (critic's guess for the next tick) - (critic's guess now)
            advantage = surprise + GAMMA * LAMBDA * (next tick's advantage)
        A death ends an agent's story, so nothing after it counts.
        Each world is worked through separately (slot numbers are per world)."""
        next_value = {k: v.copy() for k, v in bootstrap.items()}
        next_adv = {k: np.zeros(len(v), dtype=np.float32) for k, v in bootstrap.items()}
        for step in reversed(steps):
            idx, done = step["idx"], step["done"]
            next_value_w, next_adv_w = next_value[step["world"]], next_adv[step["world"]]
            nv = np.where(done, 0.0, next_value_w[idx])
            na = np.where(done, 0.0, next_adv_w[idx])
            surprise = step["rewards"] + GAMMA * nv - step["values"]
            adv = surprise + GAMMA * GAE_LAMBDA * na
            step["adv"] = adv.astype(np.float32)
            step["returns"] = (adv + step["values"]).astype(np.float32)
            next_value_w[idx] = step["values"]
            next_adv_w[idx] = adv

    # ---------------- 3. LEARN (PPO) ----------------
    def learn(self, steps):
        cat = lambda key: np.concatenate([s[key] for s in steps])
        view, feats = cat("view"), cat("feats")
        actions, old_logp = cat("actions"), cat("logp")
        adv, returns = cat("adv"), cat("returns")
        memory, slot, spawn, temper = cat("memory"), cat("idx"), cat("spawn"), cat("temper")
        world = np.concatenate([np.full(len(s["idx"]), s["world"]) for s in steps])
        which = cat("brain")
        # Scale each brain's advantages to an average of 0 and spread of 1 (steadier learning)
        adv = adv.astype(np.float32)
        for b in np.unique(which):
            rows = which == b
            adv[rows] = (adv[rows] - adv[rows].mean()) / (adv[rows].std() + 1e-8)
        layout = self.stretches(slot, spawn, world)
        start_memory = memory[layout[:, 0]]
        stretch_brain = which[layout[:, 0]]          # a whole life uses one brain
        per_minibatch = MINIBATCH // SEQ_LEN
        totals = Counter()
        updates = 0
        for _ in range(EPOCHS):
            # Each brain's stretches, shuffled and cut into groups; then the groups
            # of all brains are mixed so every brain learns a little at a time
            groups = []
            for b in np.unique(stretch_brain):
                mine = np.random.permutation(np.flatnonzero(stretch_brain == b))
                for start in range(0, len(mine), per_minibatch):
                    chosen = mine[start:start + per_minibatch]
                    if len(chosen) < per_minibatch // 4 and start > 0:
                        continue   # skip a tiny leftover group
                    groups.append((int(b), chosen))
            for k in np.random.permutation(len(groups)):
                b, chosen = groups[k]
                brain, optimizer = self.brain.brains[b], self.optimizers[b]
                lay = layout[chosen]
                filled = lay >= 0
                mb = lay[filled]                                  # the decisions in this group
                local = np.full(lay.shape, -1, dtype=np.int64)
                local[filled] = np.arange(len(mb))
                v_t = torch.from_numpy(view[mb]).to(self.device).float() / 255
                f_t = torch.from_numpy(feats[mb]).to(self.device)
                a_t = torch.from_numpy(actions[mb]).to(self.device)
                old_t = torch.from_numpy(old_logp[mb]).to(self.device)
                adv_t = torch.from_numpy(adv[mb]).to(self.device)
                ret_t = torch.from_numpy(returns[mb]).to(self.device)
                m_t = torch.from_numpy(start_memory[chosen]).to(self.device)
                lay_t = torch.from_numpy(local).to(self.device)
                t_t = torch.from_numpy(temper[mb]).to(self.device)

                new_logp, entropy, value = brain.evaluate_sequences(v_t, f_t, a_t, m_t, lay_t, t_t)
                # How much more (or less) likely is this action now than when it was taken?
                ratio = torch.exp(new_logp - old_t)
                # PPO: follow the advantage, but don't let the ratio move past 1 +/- CLIP
                policy_loss = -torch.min(ratio * adv_t,
                                         torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * adv_t).mean()
                value_loss = ((value - ret_t) ** 2).mean()
                entropy_mean = entropy.mean()
                loss = policy_loss + VALUE_WEIGHT * value_loss - ENTROPY_BONUS * entropy_mean

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(brain.parameters(), MAX_GRAD)
                optimizer.step()

                totals["policy_loss"] += policy_loss.item()
                totals["value_loss"] += value_loss.item()
                totals["entropy"] += entropy_mean.item()
                updates += 1
            if self.viewer:
                self.draw()   # keep the window responsive while learning
        return {k: v / max(updates, 1) for k, v in totals.items()}

    @staticmethod
    def stretches(slot, spawn, world=None):
        """Cut every agent's life (in this batch) into stretches of SEQ_LEN ticks.
        Returns a table (stretches x SEQ_LEN) of decision rows, -1 = padding.
        Decisions are stored tick by tick, so sorting by person (world + slot +
        spawn tick) while keeping the original order puts each life in time order."""
        n = len(slot)
        life = slot.astype(np.int64) * (1 << 32) + spawn
        if world is not None:
            life += np.asarray(world, dtype=np.int64) << 43
        order = np.argsort(life, kind="stable")
        sorted_life = life[order]
        new_life = np.r_[True, sorted_life[1:] != sorted_life[:-1]]
        life_start = np.flatnonzero(new_life)
        pos = np.arange(n) - life_start[np.cumsum(new_life) - 1]     # tick number within the life
        new_stretch = new_life | (pos % SEQ_LEN == 0)
        stretch = np.cumsum(new_stretch) - 1
        layout = np.full((stretch[-1] + 1, SEQ_LEN), -1, dtype=np.int64)
        layout[stretch, pos % SEQ_LEN] = order
        return layout

    # ---------------- 4. SAVE and LOG ----------------
    def save(self):
        extra = {"update": self.update, "total_ticks": self.total_ticks,
                 "total_samples": self.total_samples, "worlds": self.worlds, "parent": self.parent,
                 "surgeries": self.surgeries}
        save_brains(self.brain, os.path.join(self.folder, "latest.pt"), extra, self.optimizers)
        if self.update % SNAPSHOT_EVERY == 0:
            save_brains(self.brain, os.path.join(self.folder, f"brain_round{self.update:05d}.pt"), extra)

    def log(self, stats, learn_stats, seconds):
        life = np.mean(self.recent_life[-300:]) if self.recent_life else 0
        row = {"round": self.update, "world": self.worlds, "world_day": self.sim.world.day,
               "ticks": self.total_ticks, "samples": self.total_samples,
               "population": self.population(), "births": stats["births"],
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
              f"pop {self.population():4d} | survived {life:5.2f} days | "
              f"reward/day {stats['reward_per_day']:5.2f} | curiosity {stats['curiosity']:4.0%} | "
              f"births {stats['births']:3d} | deaths: {top} | {seconds:4.1f}s")

    # ---------------- main loop ----------------
    def run(self):
        try:
            while not self.stop:
                start = time.time()
                steps, bootstrap, stats = self.collect()
                if self.stop or not steps:
                    if not steps:                       # everyone everywhere has died
                        for k in range(len(self.envs)):
                            self.next_world(k)
                    continue
                self.advantages(steps, bootstrap)
                learn_stats = self.learn(steps)
                self.update += 1
                stats["curiosity"] = learn_stats["entropy"] / np.log(NUM_ACTIONS)
                self.save()
                self.log(stats, learn_stats, time.time() - start)
                if self.viewer:
                    self.viewer.training_info = self.info_lines(stats)
                for k in range(len(self.envs)):
                    if self.should_reset(k):
                        self.next_world(k)
        except KeyboardInterrupt:
            pass
        self.save()
        self.close()
        print(f"Stopped. Brain saved to {self.folder}/latest.pt")

    def should_reset(self, k=0):
        return (self.obs[k]["population"] < MIN_POPULATION
                or self.obs[k]["day"] >= MAX_WORLD_DAYS)

    def set_world(self, sim):
        """Use a new world as world 0. Everyone in it starts with an empty memory."""
        self.envs[0].sim = sim
        self.obs[0] = self.envs[0].observe()
        self.banks[0] = MemoryBank(self.brain, sim.agents.count, self.device)
        if self.viewer:
            self.viewer.set_sim(sim)

    def next_world(self, k=0):
        """Replace world k with a fresh one."""
        if k == 0:
            self.set_world(new_world())
        else:
            self.obs[k] = self.envs[k].reset()
            self.banks[k] = MemoryBank(self.brain, self.obs[k]["count"], self.device)
        self.worlds += 1

    def close(self):
        for env in self.envs:
            env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the survival brain")
    parser.add_argument("--name", default="run1", help="experiment name (folder in runs/)")
    parser.add_argument("--resume", action="store_true", help="continue from runs/<name>/latest.pt")
    parser.add_argument("--no-watch", action="store_true", help="train without the window")
    parser.add_argument("--cpu", action="store_true", help="don't use the GPU")
    parser.add_argument("--worlds", type=int, default=WORLDS,
                        help=f"worlds running at once (default {WORLDS}; each extra one uses another CPU core)")
    parser.add_argument("--brains", type=int, choices=[1, NUM_TRIBES], default=None,
                        help=f"new runs only: {NUM_TRIBES} = one brain per tribe (default), 1 = one shared brain")
    Trainer(parser.parse_args()).run()
