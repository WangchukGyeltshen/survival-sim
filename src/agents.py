"""The agents (the "people" of the world).

All agents are stored together in arrays, one slot per agent. For example
`health[7]` is the health of agent number 7. Storing them this way lets numpy
update hundreds of agents at once, which matters later when training.
"""

import numpy as np

from src.world import (GRASS, FOREST, ROCK, RIVER, POND, CAVE, DEEP_WATER,
                       CLIFF, FARMLAND, TICKS_PER_DAY, DAYS_PER_YEAR)

# ---------- actions ----------
# Each tick every agent picks one of these numbers.
REST = 0
UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4
EAT = 5
DRINK = 6
NUM_ACTIONS = 7
ACTION_NAMES = ["rest", "up", "down", "left", "right", "eat", "drink"]
MOVES = {UP: (0, -1), DOWN: (0, 1), LEFT: (-1, 0), RIGHT: (1, 0)}

MALE = 0
FEMALE = 1

# ---------- terrain rules ----------
# Tiles animals can stand on. Rock blocks movement.
ANIMAL_WALKABLE = np.zeros(9, dtype=bool)
ANIMAL_WALKABLE[[GRASS, FOREST, RIVER, POND, CAVE, FARMLAND]] = True
# Agents can also walk into deep water and off cliffs... and die.
WALKABLE = ANIMAL_WALKABLE.copy()
WALKABLE[[DEEP_WATER, CLIFF]] = True
# Tiles an agent can drink from (standing on or next to)
DRINKABLE = np.zeros(9, dtype=bool)
DRINKABLE[[RIVER, POND, DEEP_WATER]] = True

# ---------- stat rules (all stats go from 0 to 100) ----------
MAX_STAT = 100.0
FOOD_DRAIN = 0.15        # full to empty in about 7 days
WATER_DRAIN = 0.35       # full to empty in about 3 days
MOVE_ENERGY = 0.15       # energy used per step
REST_ENERGY = 0.6        # energy regained when resting
IDLE_ENERGY = 0.02       # small drain just from being awake
BERRY_FOOD = 20          # food gained from one berry
DRINK_WATER = 30         # water gained per drink
STARVE_DAMAGE = 0.5      # health lost per tick when food is empty
THIRST_DAMAGE = 0.5      # health lost per tick when water is empty
HEAL_RATE = 0.1          # health regained per tick when fed and watered
WELL_FED = 50            # food and water above this lets health recover

# ---------- cold ----------
# Warmth lost per tick (negative = warming up)
WARMTH_CHANGE = {
    ("day", False): -0.4,     # normal day: the sun warms you
    ("night", False): 0.4,
    ("day", True): 0.25,      # winter day
    ("night", True): 0.8,     # winter night
}
STORM_COLD = 0.4          # extra warmth lost in a storm
CAVE_WARMTH = 0.6         # warmth gained per tick inside a cave
FIRE_WARMTH = 1.0         # warmth gained per tick near a fire
FOREST_SHELTER = 0.7      # forests cut heat loss to 70%
HUDDLE_SHELTER = 0.2      # each agent next to you cuts heat loss by 20%...
MIN_HUDDLE = 0.4          # ...down to at most 40% of normal
COLD_DAMAGE = 0.4         # health lost per tick when warmth is empty

# ---------- other dangers ----------
POISON_FOOD = 5           # poisonous berries still fill you a little...
POISON_DAMAGE = 35        # ...but hurt a lot
POND_SICK_CHANCE = 0.3    # chance drinking pond water makes you sick
SICK_TICKS = 300          # how long sickness lasts
SICK_DAMAGE = 0.15        # health lost per tick while sick
SICK_ENERGY = 0.1         # extra energy lost per tick while sick
FLOOD_DAMAGE = 0.6        # health lost per tick standing in flood water (escapable)
FIRE_DAMAGE = 20.0        # health lost per tick standing in fire
LIFESPAN_YEARS = (3, 5)   # each agent's natural lifespan is somewhere in here
INSTANT_DEATH = 1000.0

# Causes of death (stored per agent)
ALIVE = 0
STARVATION = 1
THIRST = 2
KILLED_BY_WOLF = 3
KILLED_BY_BEAR = 4
COLD = 5
POISON = 6
SICKNESS = 7
DROWNED = 8
FELL = 9
FLOOD = 10
FIRE = 11
LIGHTNING = 12
OLD_AGE = 13
DEATH_CAUSES = ["alive", "starvation", "thirst", "wolf", "bear", "cold", "poison",
                "sickness", "drowned", "fell", "flood", "fire", "lightning", "old age"]

TICKS_PER_YEAR = TICKS_PER_DAY * DAYS_PER_YEAR


class Agents:
    def __init__(self, world, count, rng):
        self.world = world
        self.rng = rng
        self.count = count

        # Spawn on random grass or forest tiles
        land = np.argwhere(np.isin(world.terrain, (GRASS, FOREST)))
        spots = land[rng.choice(len(land), count, replace=False)]
        self.y = spots[:, 0].astype(np.int32)
        self.x = spots[:, 1].astype(np.int32)

        self.alive = np.ones(count, dtype=bool)
        self.sex = np.array([MALE, FEMALE] * (count // 2) + [MALE] * (count % 2), dtype=np.int8)
        rng.shuffle(self.sex)
        # Start as young adults, between 1 and 2 years old
        self.age = rng.integers(TICKS_PER_YEAR, 2 * TICKS_PER_YEAR, count)

        self.health = np.full(count, MAX_STAT)
        self.food = rng.uniform(60, 100, count)
        self.water = rng.uniform(60, 100, count)
        self.energy = rng.uniform(60, 100, count)
        self.warmth = np.full(count, MAX_STAT)
        self.sick = np.zeros(count, dtype=np.int32)   # ticks of sickness left
        low, high = LIFESPAN_YEARS
        self.lifespan = rng.integers(low * TICKS_PER_YEAR, high * TICKS_PER_YEAR, count)

        self.last_action = np.zeros(count, dtype=np.int8)
        self.last_hurt_by = np.zeros(count, dtype=np.int8)   # what last damaged them
        self.cause_of_death = np.zeros(count, dtype=np.int8)
        self.death_tick = np.full(count, -1)

    @property
    def population(self):
        return int(self.alive.sum())

    def hurt(self, index, amount, cause):
        """Damage one or more agents and remember what hurt them."""
        self.health[index] -= amount
        self.last_hurt_by[index] = cause

    def step(self, actions, disasters):
        """Apply one action per agent and update their stats.
        Deaths are checked later in resolve_deaths(), after animals have acted."""
        world = self.world
        a = np.asarray(actions)
        alive = self.alive
        self.last_action[alive] = a[alive]

        # Exhausted agents (no energy) cannot move, so their move becomes rest
        tired = self.energy <= 0
        moving = alive & np.isin(a, list(MOVES)) & ~tired

        # ----- movement -----
        dx = np.zeros(self.count, dtype=np.int32)
        dy = np.zeros(self.count, dtype=np.int32)
        for action, (mx, my) in MOVES.items():
            chosen = moving & (a == action)
            dx[chosen], dy[chosen] = mx, my
        new_x = np.clip(self.x + dx, 0, world.size - 1)
        new_y = np.clip(self.y + dy, 0, world.size - 1)
        can_enter = WALKABLE[world.terrain[new_y, new_x]]
        self.x = np.where(can_enter, new_x, self.x)
        self.y = np.where(can_enter, new_y, self.y)
        self.energy[moving] -= MOVE_ENERGY

        # Walking into deep water or off a cliff is deadly
        here = world.terrain[self.y, self.x]
        self.hurt(alive & (here == DEEP_WATER), INSTANT_DEATH, DROWNED)
        self.hurt(alive & (here == CLIFF), INSTANT_DEATH, FELL)

        # ----- resting -----
        resting = alive & ((a == REST) | (np.isin(a, list(MOVES)) & tired))
        self.energy[resting] += REST_ENERGY

        # ----- eating (one berry from the tile the agent stands on) -----
        # Done one agent at a time so two agents can't eat the same last berry
        for i in np.flatnonzero(alive & (a == EAT)):
            y, x = self.y[i], self.x[i]
            if world.berries[y, x] > 0:
                world.berries[y, x] -= 1
                if world.poison[y, x]:
                    self.food[i] += POISON_FOOD
                    self.hurt(i, POISON_DAMAGE, POISON)
                else:
                    self.food[i] += BERRY_FOOD

        # ----- drinking (standing on or next to water) -----
        drinking = np.flatnonzero(alive & (a == DRINK))
        if len(drinking):
            near_water = np.zeros(len(drinking), dtype=bool)
            clean = np.zeros(len(drinking), dtype=bool)
            for ox, oy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
                tx = np.clip(self.x[drinking] + ox, 0, world.size - 1)
                ty = np.clip(self.y[drinking] + oy, 0, world.size - 1)
                tile = world.terrain[ty, tx]
                near_water |= DRINKABLE[tile]
                clean |= DRINKABLE[tile] & (tile != POND)
            self.water[drinking[near_water]] += DRINK_WATER
            # Only pond water nearby: it might make you sick
            dirty = drinking[near_water & ~clean]
            unlucky = dirty[self.rng.random(len(dirty)) < POND_SICK_CHANCE]
            self.sick[unlucky] = SICK_TICKS

        # ----- body needs -----
        self.food[alive] -= FOOD_DRAIN
        self.water[alive] -= WATER_DRAIN
        self.energy[alive] -= IDLE_ENERGY
        self.age[alive] += 1

        starving = alive & (self.food <= 0)
        thirsty = alive & (self.water <= 0)
        self.hurt(starving, STARVE_DAMAGE, STARVATION)
        self.hurt(thirsty, THIRST_DAMAGE, THIRST)
        healthy = alive & (self.food > WELL_FED) & (self.water > WELL_FED) & (self.sick == 0)
        self.health[healthy] += HEAL_RATE

        # ----- sickness -----
        ill = alive & (self.sick > 0)
        self.hurt(ill, SICK_DAMAGE, SICKNESS)
        self.energy[ill] -= SICK_ENERGY
        self.sick[ill] -= 1

        # ----- cold -----
        self.update_warmth(alive, disasters)
        self.hurt(alive & (self.warmth <= 0), COLD_DAMAGE, COLD)

        # ----- floods, fire, lightning -----
        self.hurt(alive & (world.flood[self.y, self.x] > 0), FLOOD_DAMAGE, FLOOD)
        self.hurt(alive & (world.fire[self.y, self.x] > 0), FIRE_DAMAGE, FIRE)
        for lx, ly in disasters.lightning:
            self.hurt(alive & (self.x == lx) & (self.y == ly), INSTANT_DEATH, LIGHTNING)

        # ----- old age -----
        self.hurt(alive & (self.age >= self.lifespan), INSTANT_DEATH, OLD_AGE)

        for stat in (self.food, self.water, self.energy, self.warmth):
            np.clip(stat, 0, MAX_STAT, out=stat)
        np.clip(self.health, None, MAX_STAT, out=self.health)

    def update_warmth(self, alive, disasters):
        world = self.world
        time = "night" if world.is_night else "day"
        loss = np.full(self.count, WARMTH_CHANGE[(time, world.season == "winter")])
        if world.storm > 0:
            loss += STORM_COLD

        # Shelter only reduces heat loss (it doesn't stop the sun warming you)
        here = world.terrain[self.y, self.x]
        shelter = np.ones(self.count)
        shelter[here == FOREST] *= FOREST_SHELTER

        # Huddling: count other agents on the same or a touching tile (3 x 3 area)
        grid = np.zeros((world.size + 2, world.size + 2), dtype=np.int32)
        np.add.at(grid, (self.y[alive] + 1, self.x[alive] + 1), 1)
        around = sum(grid[1 + oy:world.size + 1 + oy, 1 + ox:world.size + 1 + ox]
                     for oy in (-1, 0, 1) for ox in (-1, 0, 1))
        others = around[self.y, self.x] - 1
        shelter *= np.maximum(1 - HUDDLE_SHELTER * others, MIN_HUDDLE)

        change = np.where(loss > 0, -loss * shelter, -loss)
        change[here == CAVE] = CAVE_WARMTH
        change[disasters.near_fire(self.x, self.y)] = FIRE_WARMTH
        self.warmth[alive] += change[alive]

    def resolve_deaths(self):
        """Agents with no health left die. Returns which agents died this tick."""
        dead = self.alive & (self.health <= 0)
        if dead.any():
            self.health[dead] = 0
            self.cause_of_death[dead] = self.last_hurt_by[dead]
            self.death_tick[dead] = self.world.tick
            self.alive[dead] = False
        return dead
