"""Natural disasters: storms, lightning, floods and wildfires.

They happen at random. Agents are never warned; they can only learn from
experience what a storm or a fire means.
"""

import numpy as np

from src.world import GRASS, FOREST, RIVER, POND, FARMLAND, TICKS_PER_DAY

# ----- storms -----
STORM_CHANCE = 0.0008          # per tick (about one storm every 12 days)
STORM_LENGTH = (50, 150)       # ticks
LIGHTNING_CHANCE = 0.03        # per storm tick, lightning strikes somewhere
# ----- floods -----
FLOOD_CHANCE = 0.5             # chance a storm (outside winter) causes a flood
FLOOD_REACH = 2                # tiles from rivers and ponds that flood
FLOOD_AFTER = 80               # ticks the water stays after the storm ends
# ----- wildfires -----
DRY_FIRE_CHANCE = 0.0004       # per tick in summer and autumn (no lightning needed)
BURN_TICKS = 30                # how long one tile burns
SPREAD_FOREST = 0.03           # chance per tick fire jumps to a neighbouring forest tile
SPREAD_GRASS = 0.006           # ... or grass tile
BURNT_FOREST_TICKS = 10 * TICKS_PER_DAY   # ash before berries grow back
BURNT_GRASS_TICKS = 3 * TICKS_PER_DAY

FLOODABLE = (GRASS, FOREST, FARMLAND)


def neighbours_of(mask):
    """True where a tile touches (up/down/left/right) a True tile in mask."""
    out = np.zeros_like(mask)
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


class Disasters:
    def __init__(self, world, rng):
        self.world = world
        self.rng = rng
        self.events = []        # (tick, message) for the viewer and records
        self.lightning = []     # (x, y) strikes this tick
        # Land near rivers and ponds: where floods can reach
        water = np.isin(world.terrain, (RIVER, POND))
        reach = water.copy()
        for _ in range(FLOOD_REACH):
            reach |= neighbours_of(reach)
        self.flood_zone = reach & np.isin(world.terrain, FLOODABLE)

    def log(self, message):
        self.events.append((self.world.tick, message))

    def step(self):
        w = self.world
        self.lightning = []
        winter = w.season == "winter"

        # ----- storms -----
        if w.storm > 0:
            w.storm -= 1
            if self.rng.random() < LIGHTNING_CHANCE:
                x, y = self.rng.integers(0, w.size, 2)
                self.lightning.append((int(x), int(y)))
                if w.terrain[y, x] in (GRASS, FOREST) and not winter and w.flood[y, x] == 0:
                    self.ignite(x, y, "Lightning started a fire")
            if w.storm == 0:
                self.log("The storm has passed")
        elif self.rng.random() < STORM_CHANCE:
            w.storm = int(self.rng.integers(*STORM_LENGTH))
            self.log("A blizzard begins" if winter else "A storm begins")
            if not winter and self.rng.random() < FLOOD_CHANCE:
                w.flood[self.flood_zone] = w.storm + FLOOD_AFTER
                self.log("Rivers are flooding!")

        # ----- floods drain away -----
        np.subtract(w.flood, 1, out=w.flood, where=w.flood > 0)

        # ----- wildfires -----
        dry = w.season in ("summer", "autumn") and w.storm == 0
        if dry and self.rng.random() < DRY_FIRE_CHANCE:
            forest = np.argwhere((w.terrain == FOREST) & (w.burnt == 0))
            if len(forest):
                y, x = forest[self.rng.integers(len(forest))]
                self.ignite(x, y, "A wildfire has started")
        self.spread_fire()

        # ----- ash slowly recovers -----
        np.subtract(w.burnt, 1, out=w.burnt, where=w.burnt > 0)

    def ignite(self, x, y, message):
        w = self.world
        if w.fire[y, x] == 0 and w.burnt[y, x] == 0:
            w.fire[y, x] = BURN_TICKS
            self.log(message)

    def spread_fire(self):
        w = self.world
        burning = w.fire > 0
        if not burning.any():
            return
        # Rain puts fires out faster and stops them spreading as much
        rain = w.storm > 0
        fuel = (w.fire == 0) & (w.burnt == 0) & (w.flood == 0)
        near = neighbours_of(burning) & fuel
        roll = self.rng.random(w.fire.shape)
        chance = np.where(w.terrain == FOREST, SPREAD_FOREST,
                          np.where(np.isin(w.terrain, (GRASS, FARMLAND)), SPREAD_GRASS, 0.0))
        if rain:
            chance = chance / 4
        catches = near & (roll < chance)

        # Burn down existing fires
        w.fire[burning] -= 3 if rain else 1
        np.maximum(w.fire, 0, out=w.fire)
        # Tiles that just finished burning become ash, and their berries are gone
        done = burning & (w.fire == 0)
        w.burnt[done & (w.terrain == FOREST)] = BURNT_FOREST_TICKS
        w.burnt[done & (w.terrain != FOREST)] = BURNT_GRASS_TICKS
        w.berries[burning] = 0
        # Crops burn too
        w.crop[done] = 0
        w.terrain[done & (w.terrain == FARMLAND)] = GRASS
        # Anything lying on the ground burns
        for y, x in np.argwhere(done):
            w.piles.pop((int(x), int(y)), None)
        w.fire[catches] = BURN_TICKS
        if not (w.fire > 0).any():
            self.log("The fire has burnt out")

    def near_fire(self, x, y, distance=2):
        """True for positions within `distance` tiles of a fire (warm!)."""
        w = self.world
        burning = w.fire > 0
        if not burning.any():
            return np.zeros(len(x), dtype=bool)
        warm = burning.copy()
        for _ in range(distance):
            warm |= neighbours_of(warm)
        return warm[y, x]
