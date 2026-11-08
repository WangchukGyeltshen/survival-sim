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
                       FARMLAND, TICKS_PER_DAY, MAX_BERRIES, CROP_YIELD,
                       FLINT, CLAY, COPPER_ORE, TIN_ORE, IRON_ORE, CAMPFIRE, HUT, FENCE, WALL,
                       STORAGE, BRIDGE, WELL, WATCHTOWER, FURNACE)
from src.agents import (ADULT_AGE, TICKS_PER_YEAR, ITEMS, NEW_ITEMS, CARRY_LIMIT, NUM_ACTIONS,
                        MALE, FEMALE, HANDS, GENE_NAMES, GENE_LIMITS, PREGNANCY_TICKS,
                        MAX_STAT, NUM_TRIBES, POT_SIPS, BEHAVIOUR_GENES)
from src.tech import (FLINT_BLADE, COPPER_SPEAR, BRONZE_SPEAR, IRON_SPEAR, BLUEPRINTS, NUM_CALLS)
from src.animals import RABBIT, DEER, WOLF, BEAR

VIEW_RADIUS = 7                    # longest anyone can see (vision gene is capped here)
WATCHTOWER_BONUS = 3               # extra sight when standing on a watchtower
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
    # ---- added in Stage 2 (always add new channels at the END) ----
    "flint", "clay", "copper ore", "tin ore", "iron ore",
    "campfire", "hut", "wooden fence", "stone wall", "storage", "bridge", "well", "watchtower", "furnace",
    "sick person", "own tribe person", "other tribe person",
    "call 1", "call 2", "call 3",
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
    + [f"last action {a}" for a in range(13)]
    # ---- added in Stage 2 (always add new features at the END) ----
    + [f"last action {a}" for a in range(13, NUM_ACTIONS)]
    + [f"carry {item}" for item in NEW_ITEMS]
    + ["pot water", "weapon flint blade", "weapon copper spear", "weapon bronze spear",
       "weapon iron spear", "weapon uses (long)", "pick", "sickle"]
    + [f"blueprint {k}" for k in range(len(BLUEPRINTS))]
    + [f"facing {d}" for d in ("up", "down", "left", "right")]
    + [f"heard call {k + 1}" for k in range(NUM_CALLS)]
    + [f"tribe {t}" for t in range(NUM_TRIBES)]
    + ["on watchtower", "in hut"]
    # ---- added in Stage 3 ----
    + [f"gene {g}" for g in BEHAVIOUR_GENES]
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

    for kind, name in ((FLINT, "flint"), (CLAY, "clay"), (COPPER_ORE, "copper ore"),
                       (TIN_ORE, "tin ore"), (IRON_ORE, "iron ore")):
        inner[CH[name]] = (w.deposit == kind) & (w.deposit_amount > 0)
    for kind, name in ((CAMPFIRE, "campfire"), (HUT, "hut"), (FENCE, "wooden fence"),
                       (WALL, "stone wall"), (STORAGE, "storage"), (BRIDGE, "bridge"), (WELL, "well"),
                       (WATCHTOWER, "watchtower"), (FURNACE, "furnace")):
        inner[CH[name]] = w.structure == kind

    alive = a.alive
    adult = a.age >= ADULT_AGE * TICKS_PER_YEAR
    sick = alive & (a.sick > 0)
    inner[CH["sick person"], a.y[sick], a.x[sick]] = 1
    for k in range(NUM_CALLS):
        callers = alive & (a.calling == k + 1)
        inner[CH[f"call {k + 1}"], a.y[callers], a.x[callers]] = 1
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

    # Own tribe / other tribe: depends on who is looking, so done per tribe
    for t in np.unique(a.tribe[idx]):
        tribe_map = np.zeros((2, m.shape[1], m.shape[2]), dtype=np.float32)
        r = VIEW_RADIUS
        same = a.alive & (a.tribe == t)
        other = a.alive & (a.tribe != t)
        tribe_map[0, a.y[same] + r, a.x[same] + r] = 1
        tribe_map[1, a.y[other] + r, a.x[other] + r] = 1
        rows = np.flatnonzero(a.tribe[idx] == t)
        cut = tribe_map[:, ys[rows], xs[rows]].transpose(1, 0, 2, 3)
        view[rows, CH["own tribe person"]] = cut[:, 0]
        view[rows, CH["other tribe person"]] = cut[:, 1]

    # Don't count yourself as "a person on my tile"
    c = VIEW_RADIUS
    adult = a.age[idx] >= ADULT_AGE * TICKS_PER_YEAR
    for name, mask in (("man", adult & (a.sex[idx] == MALE)),
                       ("woman", adult & (a.sex[idx] == FEMALE)),
                       ("child", ~adult),
                       ("armed person", a.weapon[idx] != HANDS),
                       ("own tribe person", np.ones(len(idx), dtype=bool)),
                       ("sick person", a.sick[idx] > 0),
                       ("call 1", a.calling[idx] == 1), ("call 2", a.calling[idx] == 2),
                       ("call 3", a.calling[idx] == 3)):
        rows = np.flatnonzero(mask)
        view[rows, CH[name], c, c] = 0

    # Blank out what's beyond each agent's eyesight (and at night, see less)
    on_tower = w.structure[a.y[idx], a.x[idx]] == WATCHTOWER
    sight = np.minimum(np.round(a.genes["vision"][idx]) + WATCHTOWER_BONUS * on_tower, VIEW_RADIUS)
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
    weapon = a.weapon[idx]
    put(np.eye(8, dtype=np.float32)[weapon][:, :4])       # the original 4 weapons
    put(np.minimum(a.weapon_uses[idx] / 40, 1))
    for g in GENE_NAMES:
        low, high = GENE_LIMITS[g]
        put((a.genes[g][idx] - low) / (high - low))
    phase = 2 * np.pi * (w.tick % TICKS_PER_DAY) / TICKS_PER_DAY
    put(np.full(n, w.is_night))
    put(np.full(n, np.sin(phase)))
    put(np.full(n, np.cos(phase)))
    put(np.tile(np.array([w.season == s for s in SEASONS], dtype=np.float32), (n, 1)))
    put(np.full(n, w.storm > 0))
    last = np.eye(NUM_ACTIONS, dtype=np.float32)[a.last_action[idx]]
    put(last[:, :13])
    # ---- Stage 2 features ----
    put(last[:, 13:])
    for item in NEW_ITEMS:
        put(a.inv[item][idx] / CARRY_LIMIT[item])
    put(a.pot_water[idx] / POT_SIPS)
    for kind in (FLINT_BLADE, COPPER_SPEAR, BRONZE_SPEAR, IRON_SPEAR):
        put(weapon == kind)
    put(a.weapon_uses[idx] / 140)
    put(a.pick[idx] / 2)
    put(a.sickle[idx] / 2)
    put(np.eye(len(BLUEPRINTS), dtype=np.float32)[a.blueprint[idx]])
    put(np.eye(4, dtype=np.float32)[a.facing[idx]])
    put(sim.agents.heard_calls()[idx])
    put(np.eye(NUM_TRIBES, dtype=np.float32)[a.tribe[idx]])
    put(on_tower_now(sim, idx))
    put(w.structure[a.y[idx], a.x[idx]] == HUT)
    for g in BEHAVIOUR_GENES:                         # -1..1 scaled to 0..1
        put((a.genes[g][idx] + 1) / 2)
    assert col == NUM_SELF
    return f


def on_tower_now(sim, idx):
    a = sim.agents
    return sim.world.structure[a.y[idx], a.x[idx]] == WATCHTOWER
