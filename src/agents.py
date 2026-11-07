"""The agents (the "people" of the world).

All agents are stored together in arrays, one slot per agent. For example
`health[7]` is the health of agent number 7. Storing them this way lets numpy
update hundreds of agents at once, which matters later when training.
"""

from collections import Counter

import numpy as np

from src.world import (GRASS, FOREST, ROCK, RIVER, POND, CAVE, DEEP_WATER,
                       CLIFF, FARMLAND, TICKS_PER_DAY, DAYS_PER_YEAR, CROP_GROW_TICKS)

# ---------- actions ----------
# Each tick every agent picks one of these numbers.
REST = 0
UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4
EAT = 5
DRINK = 6
PICK_UP = 7
PLANT = 8
CRAFT = 9
ATTACK_ANIMAL = 10
ATTACK_PERSON = 11
MATE = 12
NUM_ACTIONS = 13
ACTION_NAMES = ["rest", "up", "down", "left", "right", "eat", "drink", "pick up",
                "plant", "craft", "attack animal", "attack person", "mate"]
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

# ---------- carrying ----------
CARRY_LIMIT = {"berries": 5, "meat": 6, "seeds": 5, "sticks": 2, "stones": 2}
ITEMS = list(CARRY_LIMIT)
MEAT_FOOD = 15            # food from one piece of meat
MEAT_SPOIL_CHANCE = 0.003 # per tick, carried meat may go bad (lose one piece)
SEED_FROM_BERRY = 0.3     # chance eating a (safe) berry from a bush gives a seed
SEED_FROM_CROP = 0.5      # chance eating a crop gives a seed
CROP_FOOD = 25            # food from one serving of a ripe crop

# ---------- weapons ----------
HANDS, STICK, SHARP_STONE, SPEAR = 0, 1, 2, 3
WEAPON_NAMES = ["hands", "stick", "sharp stone", "spear"]
WEAPON_DAMAGE = [5, 9, 15, 25]
WEAPON_RANGE = [1, 1, 1, 2]       # spears reach 2 tiles
WEAPON_USES = [0, 20, 30, 40]     # hits before it breaks
ATTACK_ENERGY = 0.5
MISS_CHANCE = {0: 0.5, 1: 0.3}    # rabbits (0) and deer (1) often dodge

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

# ---------- population and breeding ----------
CAPACITY = 1200           # array slots (alive + recently dead)
POPULATION_CAP = 1000     # most agents alive at once
ADULT_AGE = 1.0           # years before an agent can breed and reaches full strength
MATE_WINDOW = 5           # after choosing "mate", an agent stays willing for this many ticks
MATE_MIN = 50             # food, water and health must all be above this to conceive
MATE_ENERGY = 2.0
PREGNANCY_TICKS = 300     # 3 days
MOTHER_REST_TICKS = 500   # a mother can't conceive again for 5 days after birth
FATHER_REST_TICKS = 100
CHILDBIRTH_RISK = 0.02    # small chance the mother dies giving birth
PREGNANT_COST = 1.5       # pregnant mothers burn 50% more food and move more slowly
CHILD_STRENGTH = 0.3      # children hit at 30% strength...
CHILD_MOVE_EVERY = 2      # ...and only manage a step every 2nd tick
NEWBORN_FOOD = 80

# ---------- genes ----------
# Each agent is born with these. Being above 1.0 (or above 5 for vision) is
# better, but costs extra food: a bigger, tougher body needs more fuel.
GENE_NAMES = ["strength", "hardiness", "endurance", "vision"]
GENE_LIMITS = {"strength": (0.5, 1.6), "hardiness": (0.5, 1.6),
               "endurance": (0.5, 1.6), "vision": (2.0, 8.0)}
GENE_START = {"strength": (1.0, 0.1), "hardiness": (1.0, 0.1),
              "endurance": (1.0, 0.1), "vision": (5.0, 0.5)}     # (average, spread)
MUTATION = {"strength": 0.05, "hardiness": 0.05, "endurance": 0.05, "vision": 0.25}
GENE_FOOD_COST = 0.5      # each +0.1 above normal adds 5% to food use

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
KICKED_BY_DEER = 14
KILLED_BY_PERSON = 15
CHILDBIRTH = 16
DEATH_CAUSES = ["alive", "starvation", "thirst", "wolf", "bear", "cold", "poison",
                "sickness", "drowned", "fell", "flood", "fire", "lightning", "old age",
                "deer", "killed by person", "childbirth"]

TICKS_PER_YEAR = TICKS_PER_DAY * DAYS_PER_YEAR


class Agents:
    def __init__(self, world, count, rng, capacity=CAPACITY):
        self.world = world
        self.rng = rng
        self.count = n = capacity    # number of slots; only some are alive

        self.x = np.zeros(n, dtype=np.int32)
        self.y = np.zeros(n, dtype=np.int32)
        self.alive = np.zeros(n, dtype=bool)
        self.used = np.zeros(n, dtype=bool)          # slot has ever held an agent
        self.sex = np.zeros(n, dtype=np.int8)
        self.age = np.zeros(n, dtype=np.int64)
        self.lifespan = np.zeros(n, dtype=np.int64)
        self.generation = np.zeros(n, dtype=np.int32)
        self.mother = np.full(n, -1, dtype=np.int32)
        self.father = np.full(n, -1, dtype=np.int32)

        self.health = np.zeros(n)
        self.prev_health = np.zeros(n)
        self.pain = np.zeros(n)                      # health lost this tick (felt as pain)
        self.food = np.zeros(n)
        self.water = np.zeros(n)
        self.energy = np.zeros(n)
        self.warmth = np.zeros(n)
        self.sick = np.zeros(n, dtype=np.int32)      # ticks of sickness left

        self.genes = {g: np.ones(n) for g in GENE_NAMES}
        self.willing = np.zeros(n, dtype=np.int64)   # willing to mate until this tick
        self.pregnant = np.zeros(n, dtype=np.int32)  # ticks until birth (0 = not pregnant)
        self.partner = np.full(n, -1, dtype=np.int32)
        self.partner_genes = {g: np.ones(n) for g in GENE_NAMES}  # father's genes, saved at conception
        self.partner_generation = np.zeros(n, dtype=np.int32)
        self.rest_until = np.zeros(n, dtype=np.int64)  # can't conceive again before this tick
        self.children = np.zeros(n, dtype=np.int32)

        # Inventory: how many of each item every agent carries
        self.inv = {item: np.zeros(n, dtype=np.int32) for item in ITEMS}
        self.inv_poison = np.zeros(n, dtype=np.int32)  # how many carried berries are poisonous
        self.weapon = np.zeros(n, dtype=np.int8)       # HANDS, STICK, SHARP_STONE or SPEAR
        self.weapon_uses = np.zeros(n, dtype=np.int32)
        self.kills = np.zeros(n, dtype=np.int32)       # people this agent has killed

        self.counts = Counter()   # how often things happen (hunts, crafts, plants...)
        self.hits = []            # (x, y) of every hit this tick, for the viewer
        self.hearts = []          # (x, y) of every conception this tick, for the viewer
        self.births = 0
        self.new_parents = []     # (mother, father) of every child born this tick

        self.last_action = np.zeros(n, dtype=np.int8)
        self.did = np.full(n, -1, dtype=np.int8)        # last action that actually worked...
        self.did_tick = np.full(n, -1000, dtype=np.int64)  # ...and when (for the viewer's icons)
        self.last_hurt_by = np.zeros(n, dtype=np.int8)   # what last damaged them
        self.cause_of_death = np.zeros(n, dtype=np.int8)
        self.death_tick = np.full(n, -1)
        self.spawn_tick = np.zeros(n, dtype=np.int64)  # when they appeared in this world

        # The first people: young adults (1 to 2 years old) on random grass or forest,
        # half male and half female, with slightly different genes
        land = np.argwhere(np.isin(world.terrain, (GRASS, FOREST)))
        spots = land[rng.choice(len(land), count, replace=False)]
        sexes = np.array([MALE, FEMALE] * (count // 2) + [MALE] * (count % 2))
        rng.shuffle(sexes)
        for k in range(count):
            genes = {g: rng.normal(*GENE_START[g]) for g in GENE_NAMES}
            age = int(rng.integers(TICKS_PER_YEAR, 2 * TICKS_PER_YEAR))
            self.add(spots[k, 1], spots[k, 0], sexes[k], genes, age=age, food=rng.uniform(60, 100))

    def free_slot(self):
        """A slot that never held anyone, or whose owner died over a day ago."""
        old = ~self.alive & (self.world.tick - self.death_tick > TICKS_PER_DAY)
        free = np.flatnonzero(~self.used | (self.used & old))
        return int(free[0]) if len(free) else None

    def add(self, x, y, sex, genes, age=0, food=NEWBORN_FOOD, generation=0, mother=-1, father=-1):
        """Put a new agent in a free slot. Returns its index (or None if full)."""
        i = self.free_slot()
        if i is None:
            return None
        rng = self.rng
        self.x[i], self.y[i] = x, y
        self.alive[i] = self.used[i] = True
        self.sex[i] = sex
        self.age[i] = age
        low, high = LIFESPAN_YEARS
        self.lifespan[i] = rng.integers(low * TICKS_PER_YEAR, high * TICKS_PER_YEAR)
        self.generation[i] = generation
        self.mother[i], self.father[i] = mother, father
        self.health[i] = self.prev_health[i] = MAX_STAT
        self.pain[i] = 0
        self.food[i] = food
        self.water[i] = rng.uniform(60, 100)
        self.energy[i] = rng.uniform(60, 100)
        self.warmth[i] = MAX_STAT
        self.sick[i] = 0
        for g in GENE_NAMES:
            self.genes[g][i] = np.clip(genes[g], *GENE_LIMITS[g])
        self.willing[i] = 0
        self.pregnant[i] = 0
        self.partner[i] = -1
        self.rest_until[i] = 0
        self.children[i] = 0
        for item in ITEMS:
            self.inv[item][i] = 0
        self.inv_poison[i] = 0
        self.weapon[i] = HANDS
        self.weapon_uses[i] = 0
        self.kills[i] = 0
        self.last_action[i] = REST
        self.did[i] = -1
        self.did_tick[i] = -1000
        self.last_hurt_by[i] = ALIVE
        self.cause_of_death[i] = ALIVE
        self.death_tick[i] = -1
        self.spawn_tick[i] = self.world.tick
        return i

    def is_adult(self, i):
        return self.age[i] >= ADULT_AGE * TICKS_PER_YEAR

    @property
    def population(self):
        return int(self.alive.sum())

    def hurt(self, index, amount, cause):
        """Damage one or more agents and remember what hurt them."""
        self.health[index] -= amount
        self.last_hurt_by[index] = cause

    def step(self, actions, disasters, animals):
        """Apply one action per agent and update their stats.
        Deaths are checked later in resolve_deaths(), after animals have acted."""
        world = self.world
        a = np.asarray(actions)
        alive = self.alive
        self.last_action[alive] = a[alive]
        self.prev_health[:] = self.health
        self.new_parents = []

        # Exhausted agents (no energy) cannot move, so their move becomes rest
        tired = self.energy <= 0
        moving = alive & np.isin(a, list(MOVES)) & ~tired
        # Children only manage a step every few ticks
        child = self.age < ADULT_AGE * TICKS_PER_YEAR
        if world.tick % CHILD_MOVE_EVERY:
            moving &= ~child

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
        move_cost = MOVE_ENERGY / self.genes["endurance"] * np.where(self.pregnant > 0, PREGNANT_COST, 1)
        self.energy[moving] -= move_cost[moving]

        # Walking into deep water or off a cliff is deadly
        here = world.terrain[self.y, self.x]
        self.hurt(alive & (here == DEEP_WATER), INSTANT_DEATH, DROWNED)
        self.hurt(alive & (here == CLIFF), INSTANT_DEATH, FELL)

        # ----- resting -----
        resting = alive & ((a == REST) | (np.isin(a, list(MOVES)) & tired))
        self.energy[resting] += REST_ENERGY

        # ----- the other actions, one agent at a time (in random order, so
        # nobody always gets the last berry or the first strike) -----
        self.hits = []
        order = np.flatnonzero(alive & (a >= EAT))
        self.rng.shuffle(order)
        for i in order:
            if self.health[i] <= 0:
                continue   # killed earlier this tick
            act = a[i]
            if act == EAT:
                self.eat(i)
            elif act == PICK_UP:
                self.pick_up(i)
            elif act == PLANT:
                self.plant(i)
            elif act == CRAFT:
                self.craft(i)
            elif act == ATTACK_ANIMAL:
                self.attack_animal(i, animals)
            elif act == ATTACK_PERSON:
                self.attack_person(i)
            elif act == MATE:
                self.want_to_mate(i)
        self.hearts = []
        self.find_partners()
        self.pregnancies()

        # Carried meat slowly goes bad
        has_meat = np.flatnonzero(alive & (self.inv["meat"] > 0))
        spoils = has_meat[self.rng.random(len(has_meat)) < MEAT_SPOIL_CHANCE]
        self.inv["meat"][spoils] -= 1

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
            self.did[drinking[near_water]] = DRINK
            self.did_tick[drinking[near_water]] = world.tick
            # Only pond water nearby: it might make you sick
            dirty = drinking[near_water & ~clean]
            unlucky = dirty[self.rng.random(len(dirty)) < POND_SICK_CHANCE]
            self.sick[unlucky] = SICK_TICKS

        # ----- body needs -----
        self.food[alive] -= self.food_use()[alive]
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

    def food_use(self):
        """Food burned per tick: more for 'expensive' genes and for pregnancy."""
        extra = sum(np.maximum(self.genes[g] - 1, 0) for g in ("strength", "hardiness", "endurance"))
        extra += np.maximum(self.genes["vision"] - 5, 0) / 5
        use = FOOD_DRAIN * (1 + GENE_FOOD_COST * extra)
        return use * np.where(self.pregnant > 0, PREGNANT_COST, 1)

    # ================= breeding =================
    def want_to_mate(self, i):
        if self.is_adult(i):
            self.willing[i] = self.world.tick + MATE_WINDOW

    def can_conceive(self, i):
        return (self.alive[i] and self.health[i] > MATE_MIN and self.food[i] > MATE_MIN
                and self.water[i] > MATE_MIN and self.is_adult(i)
                and self.pregnant[i] == 0 and self.rest_until[i] <= self.world.tick)

    def find_partners(self):
        """A willing woman and a willing man standing next to each other may conceive."""
        tick = self.world.tick
        pending = int((self.pregnant > 0).sum())
        women = np.flatnonzero(self.alive & (self.sex == FEMALE) & (self.willing >= tick))
        men = np.flatnonzero(self.alive & (self.sex == MALE) & (self.willing >= tick))
        if len(women) == 0 or len(men) == 0:
            return
        for f in women:
            if self.population + pending >= POPULATION_CAP:
                return
            if not self.can_conceive(f):
                continue
            d = np.abs(self.x[men] - self.x[f]) + np.abs(self.y[men] - self.y[f])
            for m in men[np.argsort(d)]:
                if abs(self.x[m] - self.x[f]) + abs(self.y[m] - self.y[f]) > 1:
                    break
                if self.can_conceive(m):
                    self.pregnant[f] = PREGNANCY_TICKS
                    self.partner[f] = m
                    for g in GENE_NAMES:
                        self.partner_genes[g][f] = self.genes[g][m]
                    self.partner_generation[f] = self.generation[m]
                    self.rest_until[m] = tick + FATHER_REST_TICKS
                    self.willing[f] = self.willing[m] = 0
                    self.energy[[f, m]] -= MATE_ENERGY
                    self.hearts.append((int(self.x[f]), int(self.y[f])))
                    self.counts["conceived child"] += 1
                    self.success(f, MATE)
                    self.success(m, MATE)
                    pending += 1
                    break

    def pregnancies(self):
        expecting = np.flatnonzero(self.alive & (self.pregnant > 0))
        self.pregnant[expecting] -= 1
        for f in expecting:
            if self.pregnant[f] == 0:
                self.give_birth(f)

    def give_birth(self, f):
        m = self.partner[f]
        genes = {}
        for g in GENE_NAMES:
            # Half from each parent (the father's genes were saved at conception,
            # so this works even if he has died since), plus a small random mutation
            genes[g] = (self.genes[g][f] + self.partner_genes[g][f]) / 2 + \
                self.rng.normal(0, MUTATION[g])
        generation = max(self.generation[f], self.partner_generation[f]) + 1
        child = self.add(self.x[f], self.y[f], self.rng.integers(2), genes,
                         generation=generation, mother=f, father=m)
        self.rest_until[f] = self.world.tick + MOTHER_REST_TICKS
        self.partner[f] = -1
        if child is not None:
            self.children[f] += 1
            if m >= 0 and self.alive[m]:
                self.children[m] += 1
            self.births += 1
            self.counts["child born"] += 1
            self.new_parents.append((int(f), int(m)))
        if self.rng.random() < CHILDBIRTH_RISK:
            self.hurt(f, INSTANT_DEATH, CHILDBIRTH)

    def success(self, i, action):
        """Remember that agent i's action worked (shown as an icon in the viewer)."""
        self.did[i] = action
        self.did_tick[i] = self.world.tick

    # ================= actions =================
    def eat(self, i):
        """Eat from the tile (bush or ripe crop), otherwise something carried."""
        before = self.food[i]
        self._eat(i)
        if self.food[i] > before:
            self.success(i, EAT)

    def _eat(self, i):
        w = self.world
        y, x = self.y[i], self.x[i]
        if w.berries[y, x] > 0:
            w.berries[y, x] -= 1
            if w.terrain[y, x] == FARMLAND:
                self.food[i] += CROP_FOOD
                self.give_seed(i, SEED_FROM_CROP)
                self.counts["ate crop"] += 1
            elif w.poison[y, x]:
                self.food[i] += POISON_FOOD
                self.hurt(i, POISON_DAMAGE, POISON)
            else:
                self.food[i] += BERRY_FOOD
                self.give_seed(i, SEED_FROM_BERRY)
        elif self.inv["meat"][i] > 0:
            self.inv["meat"][i] -= 1
            self.food[i] += MEAT_FOOD
            self.counts["ate meat"] += 1
        elif self.inv["berries"][i] > 0:
            # Carried berries all look alike; a poisonous one may come out
            poisoned = self.rng.random() < self.inv_poison[i] / self.inv["berries"][i]
            self.inv["berries"][i] -= 1
            if poisoned:
                self.inv_poison[i] -= 1
                self.food[i] += POISON_FOOD
                self.hurt(i, POISON_DAMAGE, POISON)
            else:
                self.food[i] += BERRY_FOOD

    def give_seed(self, i, chance):
        if self.rng.random() < chance and self.inv["seeds"][i] < CARRY_LIMIT["seeds"]:
            self.inv["seeds"][i] += 1

    def room(self, i, item):
        return CARRY_LIMIT[item] - self.inv[item][i]

    def near_rock(self, i):
        w = self.world
        for ox, oy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            x, y = self.x[i] + ox, self.y[i] + oy
            if 0 <= x < w.size and 0 <= y < w.size and w.terrain[y, x] in (ROCK, CLIFF, CAVE):
                return True
        return False

    def pick_up(self, i):
        """Pick up whatever is here, in this order: a pile of dropped things,
        a ripe crop, a berry, a stick (forest), a stone (next to rock)."""
        before = (sum(int(self.inv[item][i]) for item in ITEMS), int(self.weapon[i]))
        self._pick_up(i)
        if (sum(int(self.inv[item][i]) for item in ITEMS), int(self.weapon[i])) != before:
            self.success(i, PICK_UP)

    def _pick_up(self, i):
        w = self.world
        x, y = int(self.x[i]), int(self.y[i])
        if (x, y) in w.piles:
            self.take_pile(i, x, y)
        elif w.berries[y, x] > 0 and self.room(i, "berries") > 0:
            w.berries[y, x] -= 1
            self.inv["berries"][i] += 1
            if w.poison[y, x]:
                self.inv_poison[i] += 1
            if w.terrain[y, x] == FARMLAND:
                self.counts["harvested crop"] += 1
        elif w.terrain[y, x] == FOREST:
            if self.weapon[i] == HANDS:
                self.weapon[i], self.weapon_uses[i] = STICK, WEAPON_USES[STICK]
                self.counts["picked up stick"] += 1
            elif self.room(i, "sticks") > 0:
                self.inv["sticks"][i] += 1
                self.counts["picked up stick"] += 1
        elif self.near_rock(i) and self.room(i, "stones") > 0:
            self.inv["stones"][i] += 1
            self.counts["picked up stone"] += 1

    def take_pile(self, i, x, y):
        pile = self.world.piles[(x, y)]
        for item in ITEMS:
            n = min(pile.get(item, 0), self.room(i, item))
            self.inv[item][i] += n
            pile[item] = pile.get(item, 0) - n
            if item == "berries" and n:
                bad = min(pile.get("poison", 0), n)
                self.inv_poison[i] += bad
                pile["poison"] = pile.get("poison", 0) - bad
        if pile.get("weapon", HANDS) > self.weapon[i]:
            # Swap for the better weapon, leaving the old one behind
            old, old_uses = int(self.weapon[i]), int(self.weapon_uses[i])
            self.weapon[i], self.weapon_uses[i] = pile["weapon"], pile["weapon_uses"]
            pile["weapon"], pile["weapon_uses"] = old, old_uses
        self.counts["took from pile"] += 1
        self.world.clean_pile(x, y)

    def plant(self, i):
        w = self.world
        y, x = self.y[i], self.x[i]
        if self.inv["seeds"][i] > 0 and w.terrain[y, x] == GRASS and w.fire[y, x] == 0:
            self.inv["seeds"][i] -= 1
            w.terrain[y, x] = FARMLAND
            w.crop[y, x] = CROP_GROW_TICKS
            w.berries[y, x] = 0
            self.counts["planted seed"] += 1
            self.success(i, PLANT)

    def craft(self, i):
        """Sharp stone + stick = spear. A stone knocked against rock = sharp stone."""
        if self.weapon[i] == SHARP_STONE and self.inv["sticks"][i] > 0:
            self.inv["sticks"][i] -= 1
            self.weapon[i], self.weapon_uses[i] = SPEAR, WEAPON_USES[SPEAR]
            self.counts["crafted spear"] += 1
            self.success(i, CRAFT)
        elif self.weapon[i] < SHARP_STONE and self.inv["stones"][i] > 0 and self.near_rock(i):
            self.inv["stones"][i] -= 1
            if self.weapon[i] == STICK and self.room(i, "sticks") > 0:
                self.inv["sticks"][i] += 1   # keep the stick for later
            self.weapon[i], self.weapon_uses[i] = SHARP_STONE, WEAPON_USES[SHARP_STONE]
            self.counts["crafted sharp stone"] += 1
            self.success(i, CRAFT)

    def strike(self, i):
        """Damage this agent deals now, and wear down its weapon."""
        weapon = self.weapon[i]
        self.energy[i] -= ATTACK_ENERGY
        damage = WEAPON_DAMAGE[weapon] * self.genes["strength"][i]
        if not self.is_adult(i):
            damage *= CHILD_STRENGTH
        if weapon != HANDS:
            self.weapon_uses[i] -= 1
            if self.weapon_uses[i] <= 0:
                self.weapon[i] = HANDS
                self.counts["weapon broke"] += 1
        return damage

    def attack_animal(self, i, animals):
        reach = WEAPON_RANGE[self.weapon[i]]
        d = np.abs(animals.x - self.x[i]) + np.abs(animals.y - self.y[i])
        targets = np.flatnonzero(animals.alive & (d <= reach))
        if len(targets) == 0:
            return
        j = targets[np.argmin(d[targets])]
        kind = int(animals.species[j])
        damage = self.strike(i)
        if self.rng.random() < MISS_CHANCE.get(kind, 0):
            return   # it dodged
        self.hits.append((int(animals.x[j]), int(animals.y[j])))
        self.success(i, ATTACK_ANIMAL)
        meat = animals.hit_by_agent(j, damage, i, self)
        if meat:
            self.counts[f"hunted {animals.name(kind)}"] += 1
            take = min(meat, self.room(i, "meat"))
            self.inv["meat"][i] += take
            if meat > take:
                self.world.drop(int(self.x[i]), int(self.y[i]), {"meat": meat - take})

    def attack_person(self, i):
        reach = WEAPON_RANGE[self.weapon[i]]
        d = np.abs(self.x - self.x[i]) + np.abs(self.y - self.y[i])
        targets = np.flatnonzero(self.alive & (self.health > 0) & (d <= reach))
        targets = targets[targets != i]
        if len(targets) == 0:
            return
        j = self.rng.choice(targets[d[targets] == d[targets].min()])
        self.hurt(j, self.strike(i), KILLED_BY_PERSON)
        self.hits.append((int(self.x[j]), int(self.y[j])))
        self.success(i, ATTACK_PERSON)
        self.counts["attacked person"] += 1
        if self.health[j] <= 0:
            self.kills[i] += 1
            self.counts["killed person"] += 1

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

        change = np.where(loss > 0, -loss * shelter / self.genes["hardiness"], -loss)
        change[here == CAVE] = CAVE_WARMTH
        change[disasters.near_fire(self.x, self.y)] = FIRE_WARMTH
        self.warmth[alive] += change[alive]

    def resolve_deaths(self):
        """Agents with no health left die. Returns which agents died this tick."""
        self.pain = np.maximum(self.prev_health - self.health, 0)
        dead = self.alive & (self.health <= 0)
        for i in np.flatnonzero(dead):
            self.drop_everything(i)
            self.pregnant[i] = 0   # an unborn baby dies with its mother
        if dead.any():
            self.health[dead] = 0
            self.cause_of_death[dead] = self.last_hurt_by[dead]
            self.death_tick[dead] = self.world.tick
            self.alive[dead] = False
        return dead

    def drop_everything(self, i):
        """A dead agent leaves everything it carried on the ground."""
        items = {item: int(self.inv[item][i]) for item in ITEMS if self.inv[item][i]}
        if self.inv_poison[i]:
            items["poison"] = int(self.inv_poison[i])
        if self.weapon[i] != HANDS:
            items["weapon"] = int(self.weapon[i])
            items["weapon_uses"] = int(self.weapon_uses[i])
        if items:
            self.world.drop(int(self.x[i]), int(self.y[i]), items)
        for item in ITEMS:
            self.inv[item][i] = 0
        self.inv_poison[i] = 0
        self.weapon[i] = HANDS
