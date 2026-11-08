"""Animals: rabbits, deer, wolves and bears.

Animals do NOT learn. They follow simple fixed rules, like instincts:
    Rabbit, deer  graze, run away from anything dangerous, breed
    Wolf          hunts prey (and people) in packs, mostly at night
    Bear          stays near its home forest, eats berries, attacks anyone
                  who comes too close

Like agents, all animals are stored in arrays with one slot per animal.
The arrays have a fixed size (CAPACITY); a newborn takes the slot of a dead
animal.
"""

from dataclasses import dataclass

import numpy as np

from src.world import (GRASS, FOREST, TICKS_PER_DAY, DAYS_PER_YEAR, BRIDGE, HUT, CAMPFIRE,
                       BLOCKS_ANIMALS)
from src.agents import ANIMAL_WALKABLE, KILLED_BY_WOLF, KILLED_BY_BEAR, KICKED_BY_DEER

RABBIT = 0
DEER = 1
WOLF = 2
BEAR = 3
SPECIES_NAMES = ["rabbit", "deer", "wolf", "bear"]

TICKS_PER_YEAR = TICKS_PER_DAY * DAYS_PER_YEAR
DIRECTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]


@dataclass
class Species:
    start: int            # how many at the beginning
    cap: int              # most that can be alive at once
    health: float
    meat: float           # food a hunter gets from killing it
    vision: int           # how far it can see (tiles)
    move_chance: float    # chance to move each tick (speed)
    food_drain: float     # hunger per tick
    breed_chance: float   # chance per tick to have a baby when well fed
    lifespan_years: float
    damage: float = 0     # damage per bite (predators only)


SPECIES = {
    RABBIT: Species(start=150, cap=300, health=10, meat=12, vision=4, move_chance=0.9,
                    food_drain=0.25, breed_chance=0.004, lifespan_years=2),
    DEER: Species(start=60, cap=150, health=40, meat=40, vision=6, move_chance=0.8,
                  food_drain=0.2, breed_chance=0.0015, lifespan_years=4),
    WOLF: Species(start=20, cap=40, health=60, meat=30, vision=7, move_chance=0.9,
                  food_drain=0.12, breed_chance=0.002, lifespan_years=4, damage=15),
    BEAR: Species(start=6, cap=10, health=150, meat=100, vision=4, move_chance=0.4,
                  food_drain=0.08, breed_chance=0.0002, lifespan_years=6, damage=35),
}
CAPACITY = sum(s.cap for s in SPECIES.values())

GRAZE_FOOD = 1.5          # food prey gain per tick standing on grass or forest
PREY_ALERTNESS = 0.6      # chance prey notices danger each tick (else keeps grazing)
WOLF_SPRINT = 1.0         # wolves always move when chasing at night
PACK_SHARE = 0.5          # packmates within PACK_SHARE_RANGE get this share of a kill
PACK_SHARE_RANGE = 4
WOLF_LUNGE_RANGE = 3      # wolves take 2 steps when prey is this close
PACK_ROAM_DISTANCE = 20   # how far a hungry pack travels to look for prey
MAX_PACK = 8              # bigger packs split: a newborn starts its own pack
MIGRATE_CHANCE = 0.002    # per tick, if a species has died out, a few arrive from the edge
MIGRANTS = {RABBIT: 10, DEER: 5, WOLF: 4, BEAR: 1}
WOLF_PACK_SIZE = 4
WOLF_DAY_SPEED = 0.4      # wolves are sleepy in the day
WOLF_HUNGRY = 80          # wolves hunt when food is below this
WOLF_DAY_ATTACKS_PEOPLE = 30   # in daytime, only attack people when this hungry
BEAR_HOME_RADIUS = 6      # bears wander this far from home
BEAR_ANGER_RANGE = 2      # bears charge people this close
BEAR_BERRY_FOOD = 15
FIRE_FEAR_RANGE = 3       # wolves and bears won't come this close to a campfire
DEER_MIGRATION = 0.25     # chance per calm tick that a deer heads toward its season's grazing
MEAT_PIECE = 10           # one piece of meat an agent carries = 10 food points of the animal
DEER_KICK_CHANCE = 0.4    # a hurt deer may kick back
DEER_KICK = 8


class Animals:
    def __init__(self, world, rng):
        self.world = world
        self.rng = rng
        n = CAPACITY
        self.alive = np.zeros(n, dtype=bool)
        self.species = np.zeros(n, dtype=np.int8)
        self.x = np.zeros(n, dtype=np.int32)
        self.y = np.zeros(n, dtype=np.int32)
        self.health = np.zeros(n)
        self.food = np.zeros(n)
        self.age = np.zeros(n, dtype=np.int64)
        self.pack = np.full(n, -1, dtype=np.int32)    # wolves: which pack
        self.home_x = np.zeros(n, dtype=np.int32)     # bears: home spot
        self.home_y = np.zeros(n, dtype=np.int32)
        self.next_pack = 0
        self.fire_zone = None      # tiles near a campfire (predators keep away)
        self.fearing_fire = False
        self.pack_target = {}   # pack id -> (x, y) the pack is roaming toward
        self.killed = {name: 0 for name in SPECIES_NAMES}   # deaths by being eaten

        grass = np.argwhere(world.terrain == GRASS)
        forest = np.argwhere(world.terrain == FOREST)
        for kind, sp in SPECIES.items():
            if kind == WOLF:
                # Wolves start in packs, each pack together in one spot
                for _ in range(sp.start // WOLF_PACK_SIZE):
                    y, x = grass[rng.integers(len(grass))]
                    pack = self.next_pack
                    self.next_pack += 1
                    for _ in range(WOLF_PACK_SIZE):
                        self.spawn(WOLF, x, y, adult=True, pack=pack)
            else:
                places = forest if kind == BEAR else grass
                for _ in range(sp.start):
                    y, x = places[rng.integers(len(places))]
                    self.spawn(kind, x, y, adult=True)

    # ---------- helpers ----------
    def spawn(self, kind, x, y, adult=False, pack=-1):
        free = np.flatnonzero(~self.alive)
        if len(free) == 0 or self.count(kind) >= SPECIES[kind].cap:
            return None
        i = free[0]
        sp = SPECIES[kind]
        self.alive[i] = True
        self.species[i] = kind
        self.x[i], self.y[i] = x, y
        self.health[i] = sp.health
        self.food[i] = self.rng.uniform(60, 100)
        lifespan = sp.lifespan_years * TICKS_PER_YEAR
        self.age[i] = self.rng.integers(0, lifespan // 2) if adult else 0
        self.pack[i] = pack
        self.home_x[i], self.home_y[i] = x, y
        return i

    def count(self, kind):
        return int((self.alive & (self.species == kind)).sum())

    def walkable(self, x, y):
        size = self.world.size
        w = self.world
        if not (0 <= x < size and 0 <= y < size):
            return False
        structure = w.structure[y, x]
        if self.fearing_fire and self.fire_zone is not None and self.fire_zone[y, x]:
            return False
        return ((ANIMAL_WALKABLE[w.terrain[y, x]] or structure == BRIDGE) and not BLOCKS_ANIMALS[structure]
                and w.fire[y, x] == 0 and w.flood[y, x] == 0)

    def move_toward(self, i, tx, ty):
        """Take the step that gets closest to (tx, ty).
        If blocked (rock, lake), sidestep randomly instead. Returns False if blocked."""
        best, best_d = None, abs(self.x[i] - tx) + abs(self.y[i] - ty)
        for dx, dy in DIRECTIONS:
            nx, ny = self.x[i] + dx, self.y[i] + dy
            d = abs(nx - tx) + abs(ny - ty)
            if d < best_d and self.walkable(nx, ny):
                best, best_d = (nx, ny), d
        if best:
            self.x[i], self.y[i] = best
            return True
        self.wander(i)
        return False

    def move_away(self, i, threats_x, threats_y):
        """Take the step that gets furthest from the nearest threat."""
        def safety(x, y):
            return np.min(np.abs(threats_x - x) + np.abs(threats_y - y))
        best, best_d = None, safety(self.x[i], self.y[i])
        for dx, dy in DIRECTIONS:
            nx, ny = self.x[i] + dx, self.y[i] + dy
            if self.walkable(nx, ny):
                d = safety(nx, ny)
                if d > best_d:
                    best, best_d = (nx, ny), d
        if best:
            self.x[i], self.y[i] = best

    def wander(self, i):
        dx, dy = DIRECTIONS[self.rng.integers(4)]
        nx, ny = self.x[i] + dx, self.y[i] + dy
        if self.walkable(nx, ny):
            self.x[i], self.y[i] = nx, ny

    def nearest(self, i, xs, ys, within):
        """Index (into xs/ys) of the closest point within range, or None."""
        if len(xs) == 0:
            return None
        d = np.abs(xs - self.x[i]) + np.abs(ys - self.y[i])
        j = int(np.argmin(d))
        return j if d[j] <= within else None

    def die(self, i):
        self.alive[i] = False

    @staticmethod
    def name(kind):
        return SPECIES_NAMES[kind]

    def hit_by_agent(self, j, damage, attacker, agents):
        """An agent hits animal j. Wolves and bears fight back hard, deer may kick.
        Returns pieces of meat if the animal died, else 0."""
        kind = self.species[j]
        self.health[j] -= damage
        if self.health[j] <= 0:
            self.die(j)
            return max(1, int(round(SPECIES[kind].meat / MEAT_PIECE)))
        if kind == WOLF:
            agents.hurt(attacker, SPECIES[WOLF].damage, KILLED_BY_WOLF)
            self.food[j] = min(self.food[j], WOLF_HUNGRY - 1)   # now it's hunting you
        elif kind == BEAR:
            agents.hurt(attacker, SPECIES[BEAR].damage, KILLED_BY_BEAR)
        elif kind == DEER and self.rng.random() < DEER_KICK_CHANCE:
            agents.hurt(attacker, DEER_KICK, KICKED_BY_DEER)
        return 0

    # ---------- one tick ----------
    def step(self, agents):
        world = self.world
        night = world.is_night
        winter = world.season == "winter"

        # Tiles close to a campfire: predators won't go there or attack anyone there
        fires = world.structure == CAMPFIRE
        if fires.any():
            zone = fires.copy()
            for _ in range(FIRE_FEAR_RANGE):
                grown = zone.copy()
                grown[1:, :] |= zone[:-1, :]
                grown[:-1, :] |= zone[1:, :]
                grown[:, 1:] |= zone[:, :-1]
                grown[:, :-1] |= zone[:, 1:]
                zone = grown
            self.fire_zone = zone
        else:
            self.fire_zone = None

        people = np.flatnonzero(agents.alive & (agents.health > 0))
        px, py = agents.x[people], agents.y[people]
        # People that predators can reach: not inside a hut, not by a campfire
        safe = world.structure[agents.y[people], agents.x[people]] == HUT
        if self.fire_zone is not None:
            safe |= self.fire_zone[agents.y[people], agents.x[people]]
        self.reachable_people = people[~safe]
        # Positions of predators, worked out once per tick (used by prey)
        hunters = np.flatnonzero(self.alive & np.isin(self.species, (WOLF, BEAR)))
        self.danger_x = np.concatenate([px, self.x[hunters]])
        self.danger_y = np.concatenate([py, self.y[hunters]])
        self.prey_ids = np.flatnonzero(self.alive & np.isin(self.species, (RABBIT, DEER)))

        order = np.flatnonzero(self.alive)
        self.rng.shuffle(order)
        for i in order:
            if not self.alive[i]:
                continue   # eaten earlier this tick
            kind = self.species[i]
            sp = SPECIES[kind]
            speed = sp.move_chance
            if kind == WOLF:
                hunting = self.food[i] < WOLF_HUNGRY
                speed = WOLF_SPRINT if (night and hunting) else (speed if night else WOLF_DAY_SPEED)
            if self.rng.random() < speed:
                if kind in (RABBIT, DEER):
                    self.act_prey(i, sp, px, py, winter)
                else:
                    self.fearing_fire = True
                    if kind == WOLF:
                        self.act_wolf(i, sp, agents, self.reachable_people, night)
                    else:
                        self.act_bear(i, sp, agents, self.reachable_people)
                    self.fearing_fire = False

        self.live_and_breed()

    def act_prey(self, i, sp, px, py, winter):
        # Danger = people and predators that are close (if the animal notices)
        tx, ty = self.danger_x, self.danger_y
        if self.rng.random() < PREY_ALERTNESS and self.nearest(i, tx, ty, sp.vision) is not None:
            near = np.abs(tx - self.x[i]) + np.abs(ty - self.y[i]) <= sp.vision
            self.move_away(i, tx[near], ty[near])
            return
        # Calm: graze, and wander now and then
        if self.world.terrain[self.y[i], self.x[i]] in (GRASS, FOREST):
            self.food[i] += GRAZE_FOOD * (0.4 if winter else 1)
        if self.species[i] == DEER and self.rng.random() < DEER_MIGRATION:
            # Herds drift north in summer and south in winter
            size = self.world.size
            band = {"summer": 0.2, "winter": 0.8}.get(self.world.season, 0.5)
            target_y = int(band * size)
            if abs(self.y[i] - target_y) > size // 8:
                self.move_toward(i, self.x[i], target_y)
                return
        if self.rng.random() < 0.3:
            self.wander(i)

    def act_wolf(self, i, sp, agents, people, night):
        if self.food[i] < WOLF_HUNGRY:
            # Possible targets: prey animals, and people if night or very hungry
            prey = self.prey_ids[self.alive[self.prey_ids]]
            targets = [(self.x[prey], self.y[prey], prey, False)]
            if night or self.food[i] < WOLF_DAY_ATTACKS_PEOPLE:
                targets.append((agents.x[people], agents.y[people], people, True))
            best = None
            for xs, ys, ids, is_person in targets:
                j = self.nearest(i, xs, ys, sp.vision)
                if j is not None:
                    d = abs(xs[j] - self.x[i]) + abs(ys[j] - self.y[i])
                    if best is None or d < best[0]:
                        best = (d, ids[j], is_person, xs[j], ys[j])
            if best:
                d, target, is_person, tx, ty = best
                if d > 1:
                    self.move_toward(i, tx, ty)
                    if d <= WOLF_LUNGE_RANGE:
                        self.move_toward(i, tx, ty)   # lunge: a second step
                    d = abs(tx - self.x[i]) + abs(ty - self.y[i])
                if d <= 1:
                    self.bite(i, sp, target, is_person, agents, KILLED_BY_WOLF)
                return
            # Hungry but nothing in sight: the pack roams together to a new area
            tx, ty = self.roam_target(self.pack[i])
            if abs(tx - self.x[i]) + abs(ty - self.y[i]) <= 2:
                self.pack_target.pop(self.pack[i], None)   # arrived; pick a new spot next time
            elif not self.move_toward(i, tx, ty) and self.rng.random() < 0.1:
                self.pack_target.pop(self.pack[i], None)   # blocked too long; try elsewhere
            return
        # Not hunting: stay close to the pack
        mates = np.flatnonzero(self.alive & (self.pack == self.pack[i]))
        cx, cy = int(self.x[mates].mean()), int(self.y[mates].mean())
        if abs(cx - self.x[i]) + abs(cy - self.y[i]) > 3:
            self.move_toward(i, cx, cy)
        elif self.rng.random() < 0.5:
            self.wander(i)

    def roam_target(self, pack):
        if pack not in self.pack_target:
            mates = np.flatnonzero(self.alive & (self.pack == pack))
            cx, cy = self.x[mates].mean(), self.y[mates].mean()
            angle = self.rng.uniform(0, 2 * np.pi)
            size = self.world.size
            tx = int(np.clip(cx + np.cos(angle) * PACK_ROAM_DISTANCE, 0, size - 1))
            ty = int(np.clip(cy + np.sin(angle) * PACK_ROAM_DISTANCE, 0, size - 1))
            self.pack_target[pack] = (tx, ty)
        return self.pack_target[pack]

    def act_bear(self, i, sp, agents, people):
        # Anyone too close gets charged
        j = self.nearest(i, agents.x[people], agents.y[people], BEAR_ANGER_RANGE)
        if j is not None:
            target = people[j]
            if abs(agents.x[target] - self.x[i]) + abs(agents.y[target] - self.y[i]) <= 1:
                self.bite(i, sp, target, True, agents, KILLED_BY_BEAR)
            else:
                self.move_toward(i, agents.x[target], agents.y[target])
            return
        # Eat berries at home
        y, x = self.y[i], self.x[i]
        if self.food[i] < 80 and self.world.berries[y, x] > 0:
            self.world.berries[y, x] -= 1
            self.food[i] += BEAR_BERRY_FOOD
            return
        # Wander, but drift back home if too far away
        if abs(x - self.home_x[i]) + abs(y - self.home_y[i]) > BEAR_HOME_RADIUS:
            self.move_toward(i, self.home_x[i], self.home_y[i])
        else:
            self.wander(i)

    def bite(self, i, sp, target, is_person, agents, cause):
        if is_person:
            agents.hurt(target, sp.damage, cause)
            if agents.health[target] <= 0:
                self.food[i] += 50   # eats the body
        else:
            self.health[target] -= sp.damage
            if self.health[target] <= 0:
                meat = SPECIES[self.species[target]].meat
                self.food[i] += meat
                if self.species[i] == WOLF:
                    # Nearby packmates share the kill
                    mates = self.alive & (self.pack == self.pack[i])
                    mates &= np.abs(self.x - self.x[i]) + np.abs(self.y - self.y[i]) <= PACK_SHARE_RANGE
                    mates[i] = False
                    self.food[mates] += meat * PACK_SHARE
                self.killed[SPECIES_NAMES[self.species[target]]] += 1
                self.die(target)

    def live_and_breed(self):
        alive = np.flatnonzero(self.alive)
        self.age[alive] += 1
        for i in alive:
            sp = SPECIES[self.species[i]]
            self.food[i] = min(self.food[i] - sp.food_drain, 100)
            old = self.age[i] > sp.lifespan_years * TICKS_PER_YEAR
            burnt = self.world.fire[self.y[i], self.x[i]] > 0
            drowned = self.world.flood[self.y[i], self.x[i]] > 0 and self.rng.random() < 0.02
            if self.food[i] <= 0 or old or burnt or drowned:
                self.die(i)
                continue
            adult = self.age[i] > TICKS_PER_YEAR // 2
            if adult and self.food[i] > 60 and self.rng.random() < sp.breed_chance:
                self.food[i] -= 30   # having a baby costs food
                pack = self.pack[i]
                if self.species[i] == WOLF and (self.alive & (self.pack == pack)).sum() >= MAX_PACK:
                    pack = self.next_pack   # pack too big: the pup will start a new one
                    self.next_pack += 1
                self.spawn(self.species[i], self.x[i], self.y[i], pack=pack)
        self.migrate()

    def migrate(self):
        """If a species has died out, a small group sometimes wanders in from the map edge."""
        for kind, n in MIGRANTS.items():
            if self.count(kind) == 0 and self.rng.random() < MIGRATE_CHANCE:
                size = self.world.size
                for _ in range(50):   # find a walkable edge tile
                    side = self.rng.integers(4)
                    p = int(self.rng.integers(size))
                    x, y = [(p, 0), (p, size - 1), (0, p), (size - 1, p)][side]
                    if self.walkable(x, y):
                        pack = -1
                        if kind == WOLF:
                            pack = self.next_pack
                            self.next_pack += 1
                        for _ in range(n):
                            self.spawn(kind, x, y, adult=True, pack=pack)
                        break
