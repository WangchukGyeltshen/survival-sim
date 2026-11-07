"""The agents' senses: turns the world into numbers for the brain.

Each agent gets two things every tick:

1. VIEW: a square of tiles centred on itself (VIEW_SIZE x VIEW_SIZE). For every
   tile there is a stack of "channels", each answering one yes/no question,
   e.g. "is this tile river?", "is there a wolf here?". The agent is never told
   what any channel MEANS; it only sees that channel 14 is on, say.
   Tiles further away than the agent's vision gene are blanked out.

2. SELF: a list of numbers about its own body, belongings and the time.

All values are scaled to roughly 0..1, which neural networks learn best from.
"""

import numpy as np

from src.world import (GRASS, FOREST, ROCK, RIVER, POND, CAVE, DEEP_WATER, CLIFF,
                       FARMLAND, TICKS_PER_DAY, MAX_BERRIES, CROP_YIELD)
from src.agents import (ADULT_AGE, TICKS_PER_YEAR, ITEMS, CARRY_LIMIT, NUM_ACTIONS,
                        MALE, FEMALE, HANDS, GENE_NAMES, GENE_LIMITS, PREGNANCY_TICKS,
                        MAX_STAT, WEAPON_USES)
from src.animals import RABBIT, DEER, WOLF, BEAR

VIEW_RADIUS = 7                    # longest anyone can see (vision gene is capped here)
VIEW_SIZE = 2 * VIEW_RADIUS + 1    # 15 x 15 tiles

# ---------- view channels ----------
CHANNELS = [
    # terrain (one of these is on for each tile)
    "grass", "forest", "rock", "river", "pond", "cave", "deep water", "cliff", "farmland",
    "edge of world",
    # food
    "red berries", "purple berries", "crop growing", "crop ripe",
    # hazards and things on the ground
    "fire", "flood", "ash", "pile",
    # animals
    "rabbit", "deer", "wolf", "bear",
    # people
    "man", "woman", "child", "armed person",
    # which tiles the agent can actually see
    "visible",
]
NUM_CHANNELS = len(CHANNELS)
CH = {name: i for i, name in enumerate(CHANNELS)}
TERRAIN_ORDER = [GRASS, FOREST, ROCK, RIVER, POND, CAVE, DEEP_WATER, CLIFF, FARMLAND]

# ---------- self features ----------
SELF_FEATURES = (
    ["health", "food", "water", "energy", "warmth", "pain",
     "sick", "pregnant", "pregnancy progress", "recently gave birth",
     "adult", "age", "female"]
    + [f"carry {item}" for item in ITEMS]
    + ["weapon hands", "weapon stick", "weapon sharp stone", "weapon spear", "weapon uses"]
    + [f"gene {g}" for g in GENE_NAMES]
    + ["night", "time of day sin", "time of day cos",
       "spring", "summer", "autumn", "winter", "storm"]
    + [f"last action {a}" for a in range(NUM_ACTIONS)]
)
NUM_SELF = len(SELF_FEATURES)
SEASONS = ["spring", "summer", "autumn", "winter"]


def world_map(sim):
    """Build every channel for the whole map once per tick: (channels, size, size).
    The map is padded with VIEW_RADIUS tiles of 'edge of world' on every side,
    so agents near the edge still get a full-size view."""
    w, a, an = sim.world, sim.agents, sim.animals
    n, r = w.size, VIEW_RADIUS
    m = np.zeros((NUM_CHANNELS, n + 2 * r, n + 2 * r), dtype=np.float32)
    m[CH["edge of world"]] = 1
    inner = m[:, r:r + n, r:r + n]          # a "window" into m: writing here writes into m
    inner[CH["edge of world"]] = 0

    for k, kind in enumerate(TERRAIN_ORDER):
        inner[k] = w.terrain == kind
    farm = w.terrain == FARMLAND
    bush = w.terrain == FOREST
    inner[CH["red berries"]] = np.where(bush & ~w.poison, w.berries, 0) / MAX_BERRIES
    inner[CH["purple berries"]] = np.where(bush & w.poison, w.berries, 0) / MAX_BERRIES
    inner[CH["crop growing"]] = farm & (w.crop > 0)
    inner[CH["crop ripe"]] = np.where(farm, w.berries, 0) / CROP_YIELD
    inner[CH["fire"]] = w.fire > 0
    inner[CH["flood"]] = w.flood > 0
    inner[CH["ash"]] = w.burnt > 0
    for (x, y) in w.piles:
        inner[CH["pile"], y, x] = 1

    for kind, name in ((RABBIT, "rabbit"), (DEER, "deer"), (WOLF, "wolf"), (BEAR, "bear")):
        sel = an.alive & (an.species == kind)
        inner[CH[name], an.y[sel], an.x[sel]] = 1

    alive = a.alive
    adult = a.age >= ADULT_AGE * TICKS_PER_YEAR
    for mask, name in ((alive & adult & (a.sex == MALE), "man"),
                       (alive & adult & (a.sex == FEMALE), "woman"),
                       (alive & ~adult, "child"),
                       (alive & (a.weapon != HANDS), "armed person")):
        inner[CH[name], a.y[mask], a.x[mask]] = 1
    return m


# Offsets of every tile in the view, and their distance from the centre
_OFF = np.arange(-VIEW_RADIUS, VIEW_RADIUS + 1)
_DY, _DX = np.meshgrid(_OFF, _OFF, indexing="ij")
_DIST = np.abs(_DY) + np.abs(_DX)       # walking distance (diamond shape)


def observe(sim, agents_idx=None):
    """Return (view, self_features) for the given agents (default: all alive).
    view:  float32 array (N, NUM_CHANNELS, VIEW_SIZE, VIEW_SIZE)
    self:  float32 array (N, NUM_SELF)"""
    a, w = sim.agents, sim.world
    if agents_idx is None:
        agents_idx = np.flatnonzero(a.alive)
    idx = np.asarray(agents_idx)
    m = world_map(sim)

    # Cut out each agent's square: the padded map shifts everything by VIEW_RADIUS,
    # so the square for an agent at (x, y) starts at (x, y) in padded coordinates
    ys = a.y[idx, None, None] + VIEW_RADIUS + _DY      # (N, 15, 15)
    xs = a.x[idx, None, None] + VIEW_RADIUS + _DX
    view = m[:, ys, xs].transpose(1, 0, 2, 3).copy()   # (N, C, 15, 15)

    # Don't count yourself as "a person on my tile"
    c = VIEW_RADIUS
    adult = a.age[idx] >= ADULT_AGE * TICKS_PER_YEAR
    for name, mask in (("man", adult & (a.sex[idx] == MALE)),
                       ("woman", adult & (a.sex[idx] == FEMALE)),
                       ("child", ~adult),
                       ("armed person", a.weapon[idx] != HANDS)):
        rows = np.flatnonzero(mask)
        view[rows, CH[name], c, c] = 0

    # Blank out what's beyond each agent's eyesight (and at night, see less)
    sight = np.minimum(np.round(a.genes["vision"][idx]), VIEW_RADIUS)
    if w.is_night:
        sight = np.maximum(sight - 2, 1)
    visible = _DIST[None] <= sight[:, None, None]      # (N, 15, 15)
    view *= visible[:, None]
    view[:, CH["visible"]] = visible

    return view, self_features(sim, idx)


def self_features(sim, idx):
    a, w = sim.agents, sim.world
    n = len(idx)
    f = np.zeros((n, NUM_SELF), dtype=np.float32)
    col = 0

    def put(values):
        nonlocal col
        values = np.asarray(values, dtype=np.float32)
        if values.ndim == 1:
            values = values[:, None]
        f[:, col:col + values.shape[1]] = values
        col += values.shape[1]

    for stat in (a.health, a.food, a.water, a.energy, a.warmth):
        put(stat[idx] / MAX_STAT)
    put(np.minimum(a.pain[idx] / 20, 1))
    put(a.sick[idx] > 0)
    put(a.pregnant[idx] > 0)
    put(np.where(a.pregnant[idx] > 0, 1 - a.pregnant[idx] / PREGNANCY_TICKS, 0))
    put(a.rest_until[idx] > w.tick)
    put(a.age[idx] >= ADULT_AGE * TICKS_PER_YEAR)
    put(np.minimum(a.age[idx] / (5 * TICKS_PER_YEAR), 1))
    put(a.sex[idx] == FEMALE)
    for item in ITEMS:
        put(a.inv[item][idx] / CARRY_LIMIT[item])
    put(np.eye(4, dtype=np.float32)[a.weapon[idx]])
    put(a.weapon_uses[idx] / max(WEAPON_USES))
    for g in GENE_NAMES:
        low, high = GENE_LIMITS[g]
        put((a.genes[g][idx] - low) / (high - low))
    phase = 2 * np.pi * (w.tick % TICKS_PER_DAY) / TICKS_PER_DAY
    put(np.full(n, w.is_night))
    put(np.full(n, np.sin(phase)))
    put(np.full(n, np.cos(phase)))
    put(np.tile(np.array([w.season == s for s in SEASONS], dtype=np.float32), (n, 1)))
    put(np.full(n, w.storm > 0))
    put(np.eye(NUM_ACTIONS, dtype=np.float32)[a.last_action[idx]])
    assert col == NUM_SELF
    return f
