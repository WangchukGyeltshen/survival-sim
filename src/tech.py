"""Technology, building, sharing and calls: the newer abilities of agents.

This is a "mixin": a set of methods that the Agents class (src/agents.py)
includes, kept in their own file so agents.py stays readable.

Nothing here tells agents what is useful. Each action simply does the best
thing that is possible with what the agent carries and where it stands.
"""

import numpy as np

from src.world import (GRASS, FOREST, ROCK, RIVER, POND, DEEP_WATER, FARMLAND,
                       NO_DEPOSIT, FLINT, CLAY, COPPER_ORE, TIN_ORE, IRON_ORE, DEPOSIT_NAMES,
                       NO_STRUCTURE, CAMPFIRE, HUT, FENCE, WALL, STORAGE, BRIDGE, WELL,
                       WATCHTOWER, FURNACE, STRUCTURE_NAMES, STRUCTURE_LIFE, CAMPFIRE_MAX_FUEL,
                       BLOCKS_PEOPLE)

# ---------- weapons (index order never changes; better = more damage) ----------
HANDS, STICK, SHARP_STONE, SPEAR, FLINT_BLADE, COPPER_SPEAR, BRONZE_SPEAR, IRON_SPEAR = range(8)
WEAPON_NAMES = ["hands", "stick", "sharp stone", "spear", "flint blade",
                "copper spear", "bronze spear", "iron spear"]
WEAPON_DAMAGE = [5, 9, 15, 25, 18, 32, 40, 50]
WEAPON_RANGE = [1, 1, 1, 2, 1, 2, 2, 2]        # spears reach 2 tiles
WEAPON_USES = [0, 20, 30, 40, 40, 60, 90, 140]  # hits before it breaks

# ---------- tools ----------
NO_TOOL, BRONZE_TOOL, IRON_TOOL = 0, 1, 2
TOOL_NAMES = ["none", "bronze", "iron"]
SICKLE_BONUS = 2           # a sickle doubles what you get from a ripe crop

# ---------- blueprints: what "build" makes ----------
BLUEPRINTS = [CAMPFIRE, HUT, FENCE, WALL, STORAGE, BRIDGE, WELL, WATCHTOWER, FURNACE]
RECIPES = {
    CAMPFIRE: {"flint": 1, "sticks": 2},
    HUT: {"sticks": 4},
    FENCE: {"sticks": 2},
    WALL: {"stones": 2},
    STORAGE: {"sticks": 2, "clay": 1},
    BRIDGE: {"sticks": 3},
    WELL: {"stones": 3, "clay": 1},
    WATCHTOWER: {"sticks": 4, "stones": 2},
    FURNACE: {"stones": 3, "clay": 2},
}
REPAIR_WITH = {CAMPFIRE: "sticks", HUT: "sticks", FENCE: "sticks", WALL: "stones",
               STORAGE: "sticks", BRIDGE: "sticks", WELL: "stones", WATCHTOWER: "sticks",
               FURNACE: "stones"}
CAMPFIRE_FUEL_PER_STICK = 3 * 100       # one stick keeps a fire going 3 more days
BUILD_ENERGY = 2.0
DEMOLISH_FACTOR = 10                    # structure wear per point of damage
WATER_TILES = (RIVER, POND, DEEP_WATER)
LAND_TILES = (GRASS, FOREST, FARMLAND)

# ---------- calls ----------
NUM_CALLS = 3
HEARING = 6                             # tiles

DIRECTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]   # facing: up, down, left, right


class TechMixin:
    # ================= helpers =================
    def facing_tile(self, i):
        dx, dy = DIRECTIONS[self.facing[i]]
        return int(self.x[i] + dx), int(self.y[i] + dy)

    def in_world(self, x, y):
        return 0 <= x < self.world.size and 0 <= y < self.world.size

    def structure_near(self, i, kind, reach=1):
        """Is there a structure of this kind within `reach` tiles (square)?"""
        w = self.world
        x, y = int(self.x[i]), int(self.y[i])
        area = w.structure[max(0, y - reach):y + reach + 1, max(0, x - reach):x + reach + 1]
        return bool((area == kind).any())

    def has(self, i, needs):
        return all(self.inv[item][i] >= n for item, n in needs.items())

    def spend(self, i, needs):
        for item, n in needs.items():
            self.inv[item][i] -= n

    def better_weapon(self, i, weapon):
        return WEAPON_DAMAGE[weapon] > WEAPON_DAMAGE[self.weapon[i]]

    def equip(self, i, weapon, label):
        self.weapon[i], self.weapon_uses[i] = weapon, WEAPON_USES[weapon]
        self.done_craft(i, label)

    def done_craft(self, i, label):
        self.counts[f"crafted {label}"] += 1
        self.log_event("crafted", i, label if label.startswith(("a ", "an ")) else f"a {label}")
        self.success(i, self.CRAFT)

    # ================= craft =================
    def craft(self, i):
        """Make the most advanced thing possible right now, checked in this order:
        smelting, cooking, a pot, metal weapons and tools, flint blade, spear, sharp stone."""
        inv = self.inv
        fire = self.structure_near(i, CAMPFIRE)
        furnace = self.structure_near(i, FURNACE)

        # --- smelting ---
        if furnace and inv["iron ore"][i] and self.room(i, "iron"):
            inv["iron ore"][i] -= 1
            inv["iron"][i] += 1
            return self.done_craft(i, "iron (smelted)")
        if furnace and inv["copper"][i] and inv["tin ore"][i] and self.room(i, "bronze"):
            inv["copper"][i] -= 1
            inv["tin ore"][i] -= 1
            inv["bronze"][i] += 1
            return self.done_craft(i, "bronze (smelted)")
        if (fire or furnace) and inv["copper ore"][i] and self.room(i, "copper"):
            inv["copper ore"][i] -= 1
            inv["copper"][i] += 1
            return self.done_craft(i, "copper (smelted)")
        # --- cooking ---
        if fire and inv["meat"][i]:
            n = min(int(inv["meat"][i]), self.room(i, "cooked meat"))
            if n:
                inv["meat"][i] -= n
                inv["cooked meat"][i] += n
                return self.done_craft(i, "cooked meat")
        # --- pottery ---
        if fire and inv["clay"][i] and not inv["pot"][i]:
            inv["clay"][i] -= 1
            inv["pot"][i] = 1
            self.pot_water[i] = 0
            return self.done_craft(i, "pot")
        # --- metal weapons and tools (best metal first) ---
        for metal, tier, spear in (("iron", IRON_TOOL, IRON_SPEAR), ("bronze", BRONZE_TOOL, BRONZE_SPEAR),
                                   ("copper", None, COPPER_SPEAR)):
            if not inv[metal][i] or not inv["sticks"][i]:
                continue
            if self.better_weapon(i, spear):
                inv[metal][i] -= 1
                inv["sticks"][i] -= 1
                return self.equip(i, spear, WEAPON_NAMES[spear])
            if tier is not None and self.pick[i] < tier:
                inv[metal][i] -= 1
                inv["sticks"][i] -= 1
                self.pick[i] = tier
                return self.done_craft(i, f"{metal} pick")
            if tier is not None and self.sickle[i] < tier:
                inv[metal][i] -= 1
                inv["sticks"][i] -= 1
                self.sickle[i] = tier
                return self.done_craft(i, f"{metal} sickle")
        # --- stone age ---
        if self.weapon[i] in (SHARP_STONE, FLINT_BLADE) and inv["sticks"][i] and self.better_weapon(i, SPEAR):
            inv["sticks"][i] -= 1
            return self.equip(i, SPEAR, "spear")
        if inv["flint"][i] and self.better_weapon(i, FLINT_BLADE):
            if self.weapon[i] == STICK and self.room(i, "sticks"):
                inv["sticks"][i] += 1          # keep the stick for later
            inv["flint"][i] -= 1
            return self.equip(i, FLINT_BLADE, "flint blade")
        if inv["stones"][i] and self.near_rock(i) and self.better_weapon(i, SHARP_STONE):
            if self.weapon[i] == STICK and self.room(i, "sticks"):
                inv["sticks"][i] += 1
            inv["stones"][i] -= 1
            return self.equip(i, SHARP_STONE, "sharp stone")

    # ================= build =================
    def next_plan(self, i):
        self.blueprint[i] = (self.blueprint[i] + 1) % len(BLUEPRINTS)

    def build(self, i, animals):
        """Build the current blueprint on the tile the agent faces,
        or repair / refuel a structure of the same kind that is already there."""
        w = self.world
        kind = BLUEPRINTS[self.blueprint[i]]
        x, y = self.facing_tile(i)
        if not self.in_world(x, y):
            return
        existing = int(w.structure[y, x])
        name = STRUCTURE_NAMES[kind]

        if existing == kind:                       # repair or add fuel
            material = REPAIR_WITH[kind]
            if self.inv[material][i] <= 0:
                return
            self.inv[material][i] -= 1
            if kind == CAMPFIRE:
                w.structure_hp[y, x] = min(w.structure_hp[y, x] + CAMPFIRE_FUEL_PER_STICK, CAMPFIRE_MAX_FUEL)
                self.log_event("fed fire", i)
            else:
                w.structure_hp[y, x] = STRUCTURE_LIFE[kind]
                self.log_event("repaired", i, name)
            self.counts[f"repaired {name}"] += 1
            self.success(i, self.BUILD)
            return
        if existing != NO_STRUCTURE or not self.has(i, RECIPES[kind]):
            return
        ground = w.terrain[y, x]
        if kind == BRIDGE:
            if ground not in WATER_TILES:
                return
        elif ground not in LAND_TILES or w.fire[y, x] > 0:
            return
        if kind == FURNACE and not self.structure_near_tile(x, y, CAMPFIRE):
            return
        if BLOCKS_PEOPLE[kind] and self.someone_on(x, y, animals):
            return                                 # can't build a wall on top of someone
        self.spend(i, RECIPES[kind])
        self.energy[i] -= BUILD_ENERGY
        w.structure[y, x] = kind
        w.structure_hp[y, x] = CAMPFIRE_FUEL_PER_STICK if kind == CAMPFIRE else STRUCTURE_LIFE[kind]
        w.structure_tribe[y, x] = self.tribe[i]
        if kind == STORAGE:
            w.storage[(x, y)] = {}
        if kind == WELL:
            self.water_crops_near(x, y)
        if ground == FARMLAND and kind != BRIDGE:  # building over a field destroys the crop
            w.terrain[y, x] = GRASS
            w.crop[y, x] = 0
            w.berries[y, x] = 0
        self.counts[f"built {name}"] += 1
        self.log_event("built", i, name)
        self.success(i, self.BUILD)

    def structure_near_tile(self, x, y, kind):
        w = self.world
        area = w.structure[max(0, y - 1):y + 2, max(0, x - 1):x + 2]
        return bool((area == kind).any())

    def someone_on(self, x, y, animals):
        people = self.alive & (self.x == x) & (self.y == y)
        beasts = animals.alive & (animals.x == x) & (animals.y == y)
        return bool(people.any() or beasts.any())

    def water_crops_near(self, x, y, reach=3):
        w = self.world
        for yy in range(max(0, y - reach), min(w.size, y + reach + 1)):
            for xx in range(max(0, x - reach), min(w.size, x + reach + 1)):
                if abs(xx - x) + abs(yy - y) <= reach:
                    w.watered[yy, xx] = True

    def demolish(self, i, damage):
        """Attack with nobody in reach: hit the structure in front (or next to) you."""
        w = self.world
        spots = [self.facing_tile(i)] + [(int(self.x[i] + dx), int(self.y[i] + dy)) for dx, dy in DIRECTIONS]
        for x, y in spots:
            if self.in_world(x, y) and w.structure[y, x] != NO_STRUCTURE:
                w.structure_hp[y, x] -= int(damage * DEMOLISH_FACTOR)
                self.hits.append((x, y))
                self.success(i, self.ATTACK_PERSON)
                if w.structure_hp[y, x] <= 0:
                    name = STRUCTURE_NAMES[w.remove_structure(x, y)]
                    self.counts[f"destroyed {name}"] += 1
                    self.log_event("destroyed", i, name)
                return True
        return False

    # ================= give and store =================
    def give(self, i):
        """Hand one piece of food to the hungriest person next to you.
        With no one next to you on a storage tile: put your goods in the storage."""
        w = self.world
        d = np.abs(self.x - self.x[i]) + np.abs(self.y - self.y[i])
        near = np.flatnonzero(self.alive & (d <= 1) & (np.arange(self.count) != i))
        if len(near):
            j = int(near[np.argmin(self.food[near])])
            for item in ("cooked meat", "meat", "berries"):
                if self.inv[item][i] > 0 and self.room(j, item) > 0:
                    self.inv[item][i] -= 1
                    self.inv[item][j] += 1
                    if item == "berries" and self.inv_poison[i] > 0 and \
                            self.rng.random() < self.inv_poison[i] / (self.inv["berries"][i] + 1):
                        self.inv_poison[i] -= 1     # a poisonous berry was passed on
                        self.inv_poison[j] += 1
                    self.counts["gave food"] += 1
                    self.log_event("gave", i, (j, item))
                    self.success(i, self.GIVE)
                    return
            return
        x, y = int(self.x[i]), int(self.y[i])
        if w.structure[y, x] == STORAGE:
            box = w.storage.setdefault((x, y), {})
            stored = 0
            for item in self.STORABLE:
                n = int(self.inv[item][i])
                if n:
                    box[item] = box.get(item, 0) + n
                    self.inv[item][i] = 0
                    stored += n
            if self.inv_poison[i]:
                box["poison"] = box.get("poison", 0) + int(self.inv_poison[i])
                self.inv_poison[i] = 0
            if stored:
                self.counts["stored goods"] += 1
                self.log_event("stored", i, stored)
                self.success(i, self.GIVE)

    def take_from_storage(self, i, x, y):
        box = self.world.storage.get((x, y), {})
        took = 0
        for item in self.STORABLE:
            n = min(box.get(item, 0), self.room(i, item))
            if n:
                self.inv[item][i] += n
                box[item] -= n
                took += n
                if item == "berries" and box.get("poison", 0):
                    bad = min(box["poison"], n)
                    self.inv_poison[i] += bad
                    box["poison"] -= bad
        if took:
            self.counts["took from storage"] += 1
            self.log_event("took from storage", i, took)
        return took > 0

    # ================= gathering resources =================
    def gather_deposit(self, i):
        """Flint or clay lying where you stand."""
        w = self.world
        x, y = int(self.x[i]), int(self.y[i])
        kind = int(w.deposit[y, x])
        if kind in (FLINT, CLAY) and w.deposit_amount[y, x] > 0:
            item = DEPOSIT_NAMES[kind]
            if self.room(i, item) > 0:
                w.deposit_amount[y, x] -= 1
                self.inv[item][i] += 1
                self.counts[f"picked up {item}"] += 1
                self.log_event("picked up", i, item)
                return True
        return False

    def mine_ore(self, i):
        """Ore in a mountain tile next to you. Iron needs a pick."""
        w = self.world
        for dx, dy in DIRECTIONS:
            x, y = int(self.x[i] + dx), int(self.y[i] + dy)
            if not self.in_world(x, y) or w.terrain[y, x] != ROCK:
                continue
            kind = int(w.deposit[y, x])
            if kind not in (COPPER_ORE, TIN_ORE, IRON_ORE) or w.deposit_amount[y, x] <= 0:
                continue
            if kind == IRON_ORE and self.pick[i] == NO_TOOL:
                continue
            item = DEPOSIT_NAMES[kind]
            if self.room(i, item) <= 0:
                continue
            w.deposit_amount[y, x] -= 1
            if w.deposit_amount[y, x] == 0:
                w.deposit[y, x] = NO_DEPOSIT
            self.inv[item][i] += 1
            self.counts[f"mined {item}"] += 1
            self.log_event("mined", i, item)
            return True
        return False

    # ================= calls =================
    def call(self, i, which):
        self.calling[i] = which
        self.counts[f"call {which}"] += 1
        self.log_event("called", i, which)
        self.success(i, self.CALL_1 + which - 1)

    def heard_calls(self):
        """For every agent: which of the 3 calls they heard this tick (from others)."""
        heard = np.zeros((self.count, NUM_CALLS), dtype=bool)
        callers = np.flatnonzero(self.alive & (self.calling > 0))
        if len(callers) == 0:
            return heard
        for c in callers:
            d = np.abs(self.x - self.x[c]) + np.abs(self.y - self.y[c])
            hearers = self.alive & (d <= HEARING)
            hearers[c] = False
            heard[hearers, self.calling[c] - 1] = True
        return heard
