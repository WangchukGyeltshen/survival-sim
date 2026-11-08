"""Records a history run so it can be studied and replayed later.

Everything is written into one folder (experiments/<name>/):

    daily.csv        one row per day: population, births, deaths by cause,
                     what agents did, average genes, animals, farming...
    people.csv       every person who ever lived: name, parents, birth, death,
                     cause of death, children, kills, genes
    events.csv       every event from the log (births, deaths, fights, ...)
    heatmap.npy      how many ticks agents spent on each tile
    terrain.npy      the map (for drawing heatmaps)
    replay/          what's needed to rewatch the run:
        day_00000.pkl   a full copy of the world every SNAPSHOT_DAYS days
        actions_00000.npz  every action taken, in chunks between snapshots
    info.json        settings of the run (seed, brain, learning on/off)
"""

import csv
import json
import os
import pickle
from collections import Counter

import numpy as np

from src.agents import (NUM_ACTIONS, ACTION_NAMES, DEATH_CAUSES, ALL_GENES, ADULT_AGE,
                        TICKS_PER_YEAR, MALE)
from src.animals import RABBIT, DEER, WOLF, BEAR
from src.world import TICKS_PER_DAY, FARMLAND, STRUCTURE_NAMES
from src.tech import WEAPON_NAMES
from src.names import agent_name

SNAPSHOT_DAYS = 10


def save_snapshot(sim, path):
    """Save a full copy of the simulation (without the brain plugged into it)."""
    policy, sim.policy = sim.policy, None
    with open(path, "wb") as f:
        pickle.dump(sim, f, protocol=pickle.HIGHEST_PROTOCOL)
    sim.policy = policy


def load_snapshot(path):
    with open(path, "rb") as f:
        return pickle.load(f)


class Recorder:
    def __init__(self, sim, folder, info):
        self.sim = sim
        self.folder = folder
        self.replay_folder = os.path.join(folder, "replay")
        os.makedirs(self.replay_folder, exist_ok=True)
        with open(os.path.join(folder, "info.json"), "w") as f:
            json.dump(info, f, indent=2)
        np.save(os.path.join(folder, "terrain.npy"), sim.world.terrain)

        self.heatmap = np.zeros((sim.world.size, sim.world.size), dtype=np.int64)
        self.actions_chunk = []        # actions of each tick since the last snapshot
        self.chunk_start_day = 0
        self.day_counts = Counter()    # things that happened today
        self.day_actions = np.zeros(NUM_ACTIONS, dtype=np.int64)
        self.log_seen = sim.log_count

        # Every person gets a permanent id (slots are reused after deaths)
        self.person_id = {}            # slot -> id of the person in it now
        self.people = []               # list of dicts, index = id
        a = sim.agents
        for i in np.flatnonzero(a.alive):
            self.add_person(i)

        self.daily_file = open(os.path.join(folder, "daily.csv"), "w", newline="")
        self.daily = None
        self.events_file = open(os.path.join(folder, "events.csv"), "w", newline="")
        self.events = csv.writer(self.events_file)
        self.events.writerow(["tick", "day", "kind", "person", "name", "other", "other_name"])
        save_snapshot(sim, os.path.join(self.replay_folder, "day_00000.pkl"))

    # ---------- people ----------
    def add_person(self, i):
        a = self.sim.agents
        i = int(i)
        mother, father = int(a.mother[i]), int(a.father[i])
        pid = len(self.people)
        self.people.append({
            "id": pid, "name": agent_name(i, a.spawn_tick[i]),
            "sex": "male" if a.sex[i] == MALE else "female",
            "born_tick": int(a.spawn_tick[i]) - (int(a.age[i]) if a.spawn_tick[i] == 0 else 0),
            "founder": int(a.spawn_tick[i] == 0 and mother < 0),
            "mother_id": self.person_id.get(mother, -1) if mother >= 0 else -1,
            "father_id": self.person_id.get(father, -1) if father >= 0 else -1,
            "generation": int(a.generation[i]),
            "tribe": int(a.tribe[i]),
            "died_tick": -1, "cause": "", "children": 0, "kills": 0,
            **{f"gene_{g}": round(float(a.genes[g][i]), 3) for g in ALL_GENES},
        })
        self.person_id[i] = pid
        return pid

    def finish_person(self, i):
        a = self.sim.agents
        pid = self.person_id.get(int(i))
        if pid is None:
            return
        p = self.people[pid]
        p["died_tick"] = int(a.death_tick[i])
        p["cause"] = DEATH_CAUSES[a.cause_of_death[i]]
        p["children"] = int(a.children[i])
        p["kills"] = int(a.kills[i])

    def pid_name(self, slot):
        pid = self.person_id.get(int(slot), -1) if slot is not None and slot >= 0 else -1
        return pid, (self.people[pid]["name"] if pid >= 0 else "")

    # ---------- every tick ----------
    def record_tick(self, actions, acted):
        """Call after every sim.step(actions). `acted` marks the agents that
        were alive (and so chose an action) this tick."""
        sim = self.sim
        a = sim.agents
        alive = a.alive
        np.add.at(self.heatmap, (a.y[alive], a.x[alive]), 1)
        self.actions_chunk.append(np.asarray(actions, dtype=np.int8).copy())
        self.day_actions += np.bincount(np.asarray(actions)[acted], minlength=NUM_ACTIONS)[:NUM_ACTIONS]

        # New events since last tick (births first, so newborns have ids)
        count = min(sim.log_count - self.log_seen, len(sim.log))
        self.log_seen = sim.log_count
        new = [sim.log[k] for k in range(len(sim.log) - count, len(sim.log))]
        for tick, kind, agent, other in new:
            if kind == "born":
                self.add_person(agent)
        for tick, kind, agent, other in new:
            self.day_counts[kind] += 1
            if kind == "died":
                self.day_counts[f"died_{other}"] += 1
            pid, name = self.pid_name(agent)
            if isinstance(other, (int, np.integer)) and kind in ("attacked", "killed", "conceived", "born"):
                oid, oname = self.pid_name(other)
                other_text = oid
            elif kind == "gave":
                oid, oname = self.pid_name(other[0])
                other_text = f"{oid}:{other[1]}"
            else:
                oname, other_text = "", other if other is not None else ""
            self.events.writerow([tick, tick // TICKS_PER_DAY, kind, pid, name, other_text, oname])
            if kind == "died":
                self.finish_person(agent)

        # Once a day: write a row of daily numbers
        if sim.world.tick % TICKS_PER_DAY == 0:
            self.write_day()
        # Every SNAPSHOT_DAYS days: save the world and the actions since last time
        if sim.world.tick % (SNAPSHOT_DAYS * TICKS_PER_DAY) == 0:
            self.save_replay_chunk()

    def write_day(self):
        sim = self.sim
        a, an, w = sim.agents, sim.animals, sim.world
        alive = a.alive
        adult = a.age >= ADULT_AGE * TICKS_PER_YEAR
        total_actions = max(int(self.day_actions.sum()), 1)
        row = {
            "day": w.day, "season": w.season,
            "population": int(alive.sum()),
            "men": int((alive & adult & (a.sex == MALE)).sum()),
            "women": int((alive & adult & (a.sex != MALE)).sum()),
            "children": int((alive & ~adult).sum()),
            "births": self.day_counts["born"],
            "deaths": self.day_counts["died"],
            "max_generation": int(a.generation[alive].max()) if alive.any() else 0,
            "avg_age_years": round(float(a.age[alive].mean() / TICKS_PER_YEAR), 3) if alive.any() else 0,
            "armed": int((alive & (a.weapon > 0)).sum()),
            "farmland_tiles": int((w.terrain == FARMLAND).sum()),
            "rabbits": an.count(RABBIT), "deer": an.count(DEER),
            "wolves": an.count(WOLF), "bears": an.count(BEAR),
            "berries": int(w.berries.sum()),
            "storm": int(w.storm > 0), "fire_tiles": int((w.fire > 0).sum()),
        }
        for t in range(4):
            row[f"tribe_{t}"] = int((alive & (a.tribe == t)).sum())
        for k in range(1, len(STRUCTURE_NAMES)):
            row[f"built_{STRUCTURE_NAMES[k].replace(' ', '_')}"] = int((w.structure == k).sum())
        for k in range(1, len(WEAPON_NAMES)):
            row[f"armed_{WEAPON_NAMES[k].replace(' ', '_')}"] = int((alive & (a.weapon == k)).sum())
        row["pots"] = int((alive & (a.inv["pot"] > 0)).sum())
        row["picks"] = int((alive & (a.pick > 0)).sum())
        row["sickles"] = int((alive & (a.sickle > 0)).sum())
        row["sick"] = int((alive & (a.sick > 0)).sum())
        for g in ALL_GENES:
            row[f"gene_{g}"] = round(float(a.genes[g][alive].mean()), 4) if alive.any() else 0
        for kind in ("attacked", "killed", "hunted", "crafted", "planted", "harvested",
                     "conceived", "poisoned", "got sick", "looted", "built", "repaired", "destroyed",
                     "gave", "stored", "took from storage", "called", "mined", "caught sickness", "fell ill"):
            row[kind.replace(" ", "_")] = self.day_counts[kind]
        for cause in DEATH_CAUSES[1:]:
            row[f"died_{cause.replace(' ', '_')}"] = self.day_counts[f"died_{cause}"]
        for k, name in enumerate(ACTION_NAMES):
            row[f"share_{name.replace(' ', '_')}"] = round(int(self.day_actions[k]) / total_actions, 4)
        if self.daily is None:
            self.daily = csv.DictWriter(self.daily_file, fieldnames=list(row))
            self.daily.writeheader()
        self.daily.writerow(row)
        self.daily_file.flush()
        self.events_file.flush()
        self.day_counts = Counter()
        self.day_actions[:] = 0

    def save_replay_chunk(self):
        day = self.sim.world.day
        if self.actions_chunk:
            path = os.path.join(self.replay_folder, f"actions_{self.chunk_start_day:05d}.npz")
            np.savez_compressed(path, actions=np.stack(self.actions_chunk))
        save_snapshot(self.sim, os.path.join(self.replay_folder, f"day_{day:05d}.pkl"))
        self.actions_chunk = []
        self.chunk_start_day = day
        self.save_tables()

    def save_tables(self):
        np.save(os.path.join(self.folder, "heatmap.npy"), self.heatmap)
        a = self.sim.agents
        for i in np.flatnonzero(a.alive):     # keep living people's numbers up to date
            pid = self.person_id.get(int(i))
            if pid is not None:
                self.people[pid]["children"] = int(a.children[i])
                self.people[pid]["kills"] = int(a.kills[i])
        with open(os.path.join(self.folder, "people.csv"), "w", newline="") as f:
            if self.people:
                writer = csv.DictWriter(f, fieldnames=list(self.people[0]))
                writer.writeheader()
                writer.writerows(self.people)

    def close(self):
        """Save whatever hasn't been saved yet."""
        if self.actions_chunk:
            path = os.path.join(self.replay_folder, f"actions_{self.chunk_start_day:05d}.npz")
            np.savez_compressed(path, actions=np.stack(self.actions_chunk))
        self.save_tables()
        self.daily_file.close()
        self.events_file.close()
