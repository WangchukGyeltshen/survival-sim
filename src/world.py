"""The virtual world: a grid of terrain tiles, food, and the passing of time."""

import numpy as np

# Terrain types. Each tile in the grid stores one of these numbers.
GRASS = 0
FOREST = 1      # has berry bushes (food)
ROCK = 2        # mountain, cannot walk through
RIVER = 3       # clean flowing water
POND = 4        # shallow still water (later: may cause sickness)
CAVE = 5        # shelter
DEEP_WATER = 6  # lakes; walking in means drowning
CLIFF = 7       # mountain edge; walking in means falling
FARMLAND = 8    # created when agents plant seeds (not generated)

# One character per terrain type, used to print the map as text.
SYMBOLS = {GRASS: ".", FOREST: "T", ROCK: "#", RIVER: "~", POND: "o",
           CAVE: "C", DEEP_WATER: "W", CLIFF: "^", FARMLAND: "="}

# Time settings (from docs/DESIGN.md)
TICKS_PER_DAY = 100
DAYLIGHT_TICKS = 70
DAYS_PER_YEAR = 20

# Food settings
MAX_BERRIES = 5          # most berries a forest tile can hold
REGROW_CHANCE = 0.02     # chance per tick that a forest tile grows 1 berry

# How much of the land each terrain takes up (roughly)
DEEP_WATER_SHARE = 0.07  # lowest 7% of the land becomes lakes
POND_SHARE = 0.03        # next 3% becomes shallow ponds around the lakes
MOUNTAIN_SHARE = 0.10    # highest 10% becomes rock
FOREST_SHARE = 0.30      # wettest 30% of the remaining land becomes forest
RIVER_COUNT = 10
CAVE_COUNT = 14
CLIFF_CHANCE = 0.35      # chance a mountain edge tile is a cliff
POISON_SHARE = 0.12      # share of forest tiles whose berries are poisonous

# Farming
CROP_GROW_TICKS = 400    # 4 days for a planted seed to ripen (with water nearby)
DRY_GROWTH = 0.25        # crops far from water grow at a quarter of the speed
WATER_REACH = 3          # tiles from water that count as "watered"
CROP_YIELD = 6           # servings on a ripe crop tile
PILE_TICKS = 5 * TICKS_PER_DAY   # dropped things rot or blow away after 5 days

# ---------- resource deposits ----------
NO_DEPOSIT, FLINT, CLAY, COPPER_ORE, TIN_ORE, IRON_ORE = 0, 1, 2, 3, 4, 5
DEPOSIT_NAMES = ["", "flint", "clay", "copper ore", "tin ore", "iron ore"]
# (chance per suitable tile, amount, regrows?)
FLINT_SHARE, CLAY_SHARE = 0.25, 0.2          # of land tiles next to rock / next to water
COPPER_SHARE, TIN_SHARE, IRON_SHARE = 0.12, 0.07, 0.08   # of mountain edge tiles
DEPOSIT_AMOUNT = {FLINT: 3, CLAY: 4, COPPER_ORE: 6, TIN_ORE: 4, IRON_ORE: 8}
DEPOSIT_REGROW = 5 * TICKS_PER_DAY            # flint and clay slowly build up again

# ---------- structures (built by agents) ----------
NO_STRUCTURE, CAMPFIRE, HUT, FENCE, WALL, STORAGE, BRIDGE, WELL, WATCHTOWER, FURNACE = range(10)
STRUCTURE_NAMES = ["", "campfire", "hut", "wooden fence", "stone wall", "storage", "bridge",
                   "well", "watchtower", "furnace"]
STRUCTURE_LIFE = {CAMPFIRE: 3 * TICKS_PER_DAY, HUT: 30 * TICKS_PER_DAY, FENCE: 20 * TICKS_PER_DAY,
                  WALL: 100 * TICKS_PER_DAY, STORAGE: 30 * TICKS_PER_DAY, BRIDGE: 40 * TICKS_PER_DAY,
                  WELL: 100 * TICKS_PER_DAY, WATCHTOWER: 40 * TICKS_PER_DAY, FURNACE: 100 * TICKS_PER_DAY}
CAMPFIRE_MAX_FUEL = 6 * TICKS_PER_DAY
# Which structures people / animals cannot walk into
BLOCKS_PEOPLE = np.zeros(10, dtype=bool)
BLOCKS_PEOPLE[[CAMPFIRE, FENCE, WALL, WELL, FURNACE]] = True
BLOCKS_ANIMALS = BLOCKS_PEOPLE.copy()
BLOCKS_ANIMALS[HUT] = True


def smooth_noise(size, rng, octaves=5, start_cells=4):
    """Random 'hilly' values between 0 and 1.

    Works by making a few coarse random grids, stretching each one smoothly
    over the map, and adding them together. Coarse grids make big hills,
    fine grids add small bumps. This is called fractal value noise.
    """
    total = np.zeros((size, size))
    weight, weights = 1.0, 0.0
    cells = start_cells
    for _ in range(octaves):
        grid = rng.random((cells + 1, cells + 1))
        # Position of every map tile inside the coarse grid
        pos = np.linspace(0, cells, size, endpoint=False)
        i = pos.astype(int)
        f = pos - i
        f = f * f * (3 - 2 * f)  # smoothstep: softens the joins
        y0, x0 = i[:, None], i[None, :]
        fy, fx = f[:, None], f[None, :]
        top = grid[y0, x0] * (1 - fx) + grid[y0, x0 + 1] * fx
        bottom = grid[y0 + 1, x0] * (1 - fx) + grid[y0 + 1, x0 + 1] * fx
        total += (top * (1 - fy) + bottom * fy) * weight
        weights += weight
        weight *= 0.5
        cells *= 2
    total /= weights
    return (total - total.min()) / (total.max() - total.min())


class World:
    def __init__(self, size=128, seed=None):
        self.size = size
        self.seed = seed
        self.rng = np.random.default_rng(seed)  # same seed = same world
        self.tick = 0
        self.terrain = np.full((size, size), GRASS, dtype=np.int8)
        self._generate()
        # Berries only exist on forest tiles. Start every forest tile full.
        self.berries = np.where(self.terrain == FOREST, MAX_BERRIES, 0).astype(np.int8)

        # Disaster layers (see src/disasters.py). Each holds ticks remaining.
        shape = (size, size)
        self.fire = np.zeros(shape, dtype=np.int16)    # burning now
        self.burnt = np.zeros(shape, dtype=np.int16)   # ash, slowly regrowing
        self.flood = np.zeros(shape, dtype=np.int16)   # flooded by a storm
        self.storm = 0                                  # ticks of storm left

        # Farming: ticks until each planted tile is ripe (0 = not growing)
        self.crop = np.zeros(shape, dtype=np.int16)
        water = np.isin(self.terrain, (RIVER, POND, DEEP_WATER))
        self.watered = water.copy()
        for _ in range(WATER_REACH):
            grown = self.watered.copy()
            grown[1:, :] |= self.watered[:-1, :]
            grown[:-1, :] |= self.watered[1:, :]
            grown[:, 1:] |= self.watered[:, :-1]
            grown[:, :-1] |= self.watered[:, 1:]
            self.watered = grown

        # Things lying on the ground: {(x, y): {"meat": 3, "weapon": 2, ..., "tick": t}}
        self.piles = {}

        # Structures built by agents, how long until each wears out, which tribe built it,
        # and what is kept in each storage
        self.structure = np.zeros(shape, dtype=np.int8)
        self.structure_hp = np.zeros(shape, dtype=np.int32)
        self.structure_tribe = np.full(shape, -1, dtype=np.int8)
        self.storage = {}       # {(x, y): {"berries": 3, ...}}
        self.worn_out = []      # structures that fell apart this tick

    # ---------- world generation ----------
    def _generate(self):
        self.elevation = smooth_noise(self.size, self.rng)
        moisture = smooth_noise(self.size, self.rng, start_cells=6)
        e = self.elevation

        # Lakes in the lowest ground, ponds around them, mountains on the highest
        deep_level = np.quantile(e, DEEP_WATER_SHARE)
        pond_level = np.quantile(e, DEEP_WATER_SHARE + POND_SHARE)
        rock_level = np.quantile(e, 1 - MOUNTAIN_SHARE)
        self.terrain[e < pond_level] = POND
        self.terrain[e < deep_level] = DEEP_WATER
        self.terrain[e >= rock_level] = ROCK

        # Forests where the ground is wettest
        land = self.terrain == GRASS
        forest_level = np.quantile(moisture[land], 1 - FOREST_SHARE)
        self.terrain[land & (moisture >= forest_level)] = FOREST

        self._carve_rivers()
        self._mountain_edges()

        # Poisonous bushes grow in patches inside forests. They look a little
        # different (purple berries), but agents are never told what that means.
        patches = smooth_noise(self.size, self.rng, start_cells=10)
        forest = self.terrain == FOREST
        level = np.quantile(patches[forest], 1 - POISON_SHARE)
        self.poison = forest & (patches >= level)
        self._place_deposits()

    def _neighbours(self, y, x):
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < self.size and 0 <= nx < self.size:
                yield ny, nx

    def _carve_rivers(self):
        """Rivers start near mountains and flow downhill until they reach water."""
        e = self.elevation
        high = np.quantile(e, 1 - MOUNTAIN_SHARE - 0.05)
        rock_level = np.quantile(e, 1 - MOUNTAIN_SHARE)
        starts = np.argwhere((e >= high) & (e < rock_level))
        if len(starts) == 0:
            return
        for _ in range(RIVER_COUNT):
            y, x = starts[self.rng.integers(len(starts))]
            visited = set()
            for _ in range(self.size * 3):
                if self.terrain[y, x] in (DEEP_WATER, POND) or (y, x) in visited:
                    break
                visited.add((y, x))
                self.terrain[y, x] = RIVER
                options = [(ny, nx) for ny, nx in self._neighbours(y, x)
                           if (ny, nx) not in visited]
                if not options:
                    break
                # Mostly go downhill, with a little randomness so rivers wiggle
                y, x = min(options, key=lambda p: e[p] + self.rng.random() * 0.02)

    def _mountain_edges(self):
        """Some rock tiles at the edge of mountains become caves or cliffs."""
        edges = []
        for y, x in np.argwhere(self.terrain == ROCK):
            if any(self.terrain[n] in (GRASS, FOREST) for n in self._neighbours(y, x)):
                edges.append((y, x))
        self.rng.shuffle(edges)
        for i, (y, x) in enumerate(edges):
            if i < CAVE_COUNT:
                self.terrain[y, x] = CAVE
            elif self.rng.random() < CLIFF_CHANCE:
                self.terrain[y, x] = CLIFF

    # ---------- time ----------
    @property
    def day(self):
        return self.tick // TICKS_PER_DAY

    @property
    def is_night(self):
        return self.tick % TICKS_PER_DAY >= DAYLIGHT_TICKS

    @property
    def season(self):
        """Year split into 4 equal seasons."""
        day_of_year = self.day % DAYS_PER_YEAR
        return ["spring", "summer", "autumn", "winter"][day_of_year * 4 // DAYS_PER_YEAR]

    # ---------- resource deposits ----------
    def _place_deposits(self):
        size = self.size
        self.deposit = np.zeros((size, size), dtype=np.int8)
        self.deposit_amount = np.zeros((size, size), dtype=np.int16)
        rock = self.terrain == ROCK
        water = np.isin(self.terrain, (RIVER, POND, DEEP_WATER))
        land = np.isin(self.terrain, (GRASS, FOREST))
        near_rock, near_water, near_land = self._touching(rock), self._touching(water), self._touching(land)
        roll = self.rng.random((size, size))
        places = [
            (FLINT, land & near_rock, FLINT_SHARE),
            (CLAY, (self.terrain == GRASS) & near_water, CLAY_SHARE),
            (COPPER_ORE, rock & near_land, COPPER_SHARE),
            (TIN_ORE, rock & near_land, TIN_SHARE),
            (IRON_ORE, rock & near_land, IRON_SHARE),
        ]
        start = 0.0
        for kind, where, share in places:
            # different slices of the same random roll, so deposits never overlap
            chosen = where & (self.deposit == 0) & (roll >= start) & (roll < start + share)
            self.deposit[chosen] = kind
            self.deposit_amount[chosen] = DEPOSIT_AMOUNT[kind]
            start += share
        self.deposit_full = self.deposit_amount.copy()

    def _touching(self, mask):
        out = np.zeros_like(mask)
        out[1:, :] |= mask[:-1, :]
        out[:-1, :] |= mask[1:, :]
        out[:, 1:] |= mask[:, :-1]
        out[:, :-1] |= mask[:, 1:]
        return out

    # ---------- structures ----------
    def can_walk(self, x, y):
        """Arrays: can people walk onto these tiles (terrain allowing)?"""
        return ~BLOCKS_PEOPLE[self.structure[y, x]]

    def remove_structure(self, x, y):
        kind = int(self.structure[y, x])
        if kind == STORAGE and (x, y) in self.storage:
            self.drop(x, y, self.storage.pop((x, y)))
        self.structure[y, x] = NO_STRUCTURE
        self.structure_hp[y, x] = 0
        self.structure_tribe[y, x] = -1
        return kind

    def _wear_structures(self):
        built = self.structure > 0
        if not built.any():
            return
        self.structure_hp[built] -= 1
        # rain puts campfires out faster
        if self.storm > 0:
            self.structure_hp[self.structure == CAMPFIRE] -= 2
        for y, x in np.argwhere(built & (self.structure_hp <= 0)):
            self.remove_structure(int(x), int(y))
            self.worn_out.append((int(x), int(y)))

    def step(self):
        """Advance the world by one tick."""
        self.tick += 1
        self._regrow_berries()
        self._grow_crops()
        self.worn_out = []
        self._wear_structures()
        if self.tick % DEPOSIT_REGROW == 0:      # flint and clay build up again
            renew = np.isin(self.deposit, (FLINT, CLAY)) & (self.deposit_amount < self.deposit_full)
            self.deposit_amount[renew] += 1
        if self.tick % 50 == 0:
            self._clean_old_piles()

    def _grow_crops(self):
        growing = (self.terrain == FARMLAND) & (self.crop > 0)
        if self.season != "winter" and growing.any():
            # Watered crops grow every tick; dry ones only sometimes
            grows = growing & (self.watered | (self.rng.random(self.crop.shape) < DRY_GROWTH))
            self.crop[grows] -= 1
            ripe = growing & (self.crop == 0)
            self.berries[ripe] = CROP_YIELD
        # Fully eaten (or burnt) farmland turns back into grass
        empty = (self.terrain == FARMLAND) & (self.crop == 0) & (self.berries == 0)
        self.terrain[empty] = GRASS

    # ---------- things on the ground ----------
    def drop(self, x, y, items):
        pile = self.piles.setdefault((x, y), {})
        for item, n in items.items():
            if item in ("weapon", "weapon_uses", "pick", "sickle", "pot_water"):
                if n > pile.get(item, 0) or item == "weapon_uses":
                    pile[item] = n
            else:
                pile[item] = pile.get(item, 0) + n
        pile["tick"] = self.tick

    def clean_pile(self, x, y):
        """Remove a pile once nothing useful is left in it."""
        pile = self.piles.get((x, y))
        if pile is not None and not any(n > 0 for k, n in pile.items()
                                        if k not in ("tick", "poison", "weapon_uses")):
            del self.piles[(x, y)]

    def _clean_old_piles(self):
        for spot in [s for s, p in self.piles.items() if self.tick - p["tick"] > PILE_TICKS]:
            del self.piles[spot]

    def _regrow_berries(self):
        chance = 0.0 if self.season == "winter" else REGROW_CHANCE  # no growth in winter
        grows = (self.terrain == FOREST) & (self.burnt == 0) & \
            (self.rng.random(self.terrain.shape) < chance)
        self.berries = np.minimum(self.berries + grows, MAX_BERRIES).astype(np.int8)

    # ---------- display ----------
    def render_text(self):
        rows = ["".join(SYMBOLS[t] for t in row) for row in self.terrain]
        time = "night" if self.is_night else "day"
        header = f"Day {self.day}, tick {self.tick} ({time}, {self.season})"
        return header + "\n" + "\n".join(rows)
