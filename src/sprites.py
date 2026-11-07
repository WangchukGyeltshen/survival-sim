"""Pixel-art tiles, drawn in code (no image files needed).

Each tile is an 8 x 8 grid of letters. Each letter is a colour from a palette.
The tiles are then scaled up so every "pixel" becomes a chunky square.
"""

import pygame

from src.world import (GRASS, FOREST, ROCK, RIVER, POND, CAVE,
                       DEEP_WATER, CLIFF, FARMLAND)

PIXELS = 8   # each tile is 8 x 8 pixel-art pixels

# ---------- colour palettes ----------
SUMMER = {
    "G": (106, 168, 79),   # grass
    "g": (139, 196, 98),   # grass tuft
    "y": (242, 214, 85),   # flower
    "L": (47, 112, 58),    # tree leaves
    "l": (78, 150, 74),    # leaf highlight
    "d": (30, 78, 44),     # leaf shadow
    "b": (110, 74, 44),    # trunk
    "R": (128, 128, 136),  # rock
    "r": (164, 164, 172),  # rock highlight
    "k": (88, 88, 98),     # rock shadow
    "K": (28, 22, 26),     # cave darkness
    "W": (58, 128, 212),   # river
    "w": (138, 190, 240),  # river ripple
    "P": (52, 96, 120),    # pond
    "p": (92, 140, 160),   # pond ripple
    "n": (84, 150, 70),    # lily pad
    "D": (30, 66, 140),    # deep water
    "e": (54, 98, 176),    # deep water wave
    "c": (70, 58, 56),     # cliff face
    "C": (46, 36, 36),     # cliff face shadow
    "f": (122, 86, 54),    # farmland soil
    "F": (92, 62, 40),     # farmland furrow
}

WINTER = dict(SUMMER)
WINTER.update({
    "G": (222, 230, 238),  # snow
    "g": (190, 204, 220),  # snow shadow
    "y": (190, 204, 220),  # no flowers in winter
    "L": (44, 92, 70),     # darker, colder leaves
    "l": (240, 246, 250),  # snow on leaves
    "r": (230, 236, 242),  # snow on rock
    "W": (96, 150, 205),
    "P": (110, 140, 160),
    "n": (110, 140, 160),
    "D": (60, 92, 150),
    "f": (200, 196, 196),  # snow on soil
})

# ---------- tile patterns ----------
GRASS_TILES = [
    ["GGGGGGGG",
     "GGgGGGGG",
     "GgGgGGGG",
     "GGGGGGGG",
     "GGGGGgGG",
     "GGGGgGgG",
     "GGGGGGGG",
     "GGGGGGGG"],
    ["GGGGGGGG",
     "GGGGGGgG",
     "GGGGGgGG",
     "GGGGGGGG",
     "GgGGGGGG",
     "gGgGGGGG",
     "GGGGGGGG",
     "GGGGGGGG"],
    ["GGGGGGGG",
     "GGGGGGGG",
     "GGGyGGGG",
     "GGygyGGG",
     "GGGyGGGG",
     "GGGGGGgG",
     "GGGGGgGg",
     "GGGGGGGG"],
    ["GGGGGGGG",
     "GGGGGGGG",
     "GGGGGGGG",
     "GGGGGGGG",
     "GGGGGGGG",
     "GGGGGGGG",
     "GGGGGGGG",
     "GGGGGGGG"],
]

TREE = ["GGLLLLGG",
        "GLllLLLG",
        "LlLLLLLd",
        "LLLLLLLd",
        "LLLLLLdd",
        "GdLLLddG",
        "GGGbbGGG",
        "GGGbbGGG"]

# Where berries appear on the tree (row, column), one spot per berry
BERRY_SPOTS = [(1, 5), (2, 3), (3, 1), (3, 5), (5, 3)]
BERRY_COLOUR = (214, 40, 64)
POISON_COLOUR = (190, 90, 255)   # poisonous berries are purple

ROCK_TILE = ["rrRRRRRr",
             "rRRRRkRR",
             "RRRRkRRR",
             "RRRRRRRk",
             "RkRRRRRR",
             "RRkRRRrR",
             "RRRRRRRR",
             "kRRRRkRk"]

CAVE_TILE = ["rrRRRRRr",
             "rRRkkkRR",
             "RRkKKKkR",
             "RkKKKKKk",
             "RkKKKKKk",
             "RkKKKKKk",
             "RkKKKKKk",
             "kkKKKKKk"]

WATER = ["WWWWWWWW",
         "WwwWWWWW",
         "WWWWWWWW",
         "WWWWWwwW",
         "WWWWWWWW",
         "WWwwWWWW",
         "WWWWWWWW",
         "WWWWWWww"]

POND_TILE = ["PPPPPPPP",
             "PppPPPPP",
             "PPPPPPPP",
             "PPPPnnPP",
             "PPPnnnPP",
             "PPPPPPPP",
             "PPPPPppP",
             "PPPPPPPP"]


DEEP_TILE = ["DDDDDDDD",
             "DDDeeDDD",
             "DDDDDDDD",
             "DDDDDDDD",
             "eeDDDDDD",
             "DDDDDDDD",
             "DDDDDDeD",
             "DDDDDDDD"]

CLIFF_TILE = ["rrRRRRrr",
              "RRRRRRRR",
              "kkkkkkkk",
              "cCccCccC",
              "ccCcccCc",
              "CccCccCc",
              "cCccCcCC",
              "CCcCCCCC"]

FARM_TILE = ["ffffffff",
             "FFFFFFFF",
             "ffffffff",
             "ffffffff",
             "FFFFFFFF",
             "ffffffff",
             "ffffffff",
             "FFFFFFFF"]


def shift(pattern, amount):
    """Slide each row sideways (used to animate water ripples)."""
    return [row[-amount:] + row[:-amount] for row in pattern]


class SpriteSheet:
    """Turns the letter patterns into scaled-up pygame images, cached for speed."""

    def __init__(self, tile_px):
        self.tile_px = tile_px
        self.scale = tile_px // PIXELS
        self.cache = {}

    def _build(self, pattern, palette):
        small = pygame.Surface((PIXELS, PIXELS))
        for y, row in enumerate(pattern):
            for x, ch in enumerate(row):
                small.set_at((x, y), palette[ch])
        # Plain scaling keeps the hard pixel edges (no blurring)
        return pygame.transform.scale(small, (self.tile_px, self.tile_px))

    def get(self, kind, variant, winter, frame):
        key = (kind, variant, winter, frame)
        if key not in self.cache:
            palette = WINTER if winter else SUMMER
            if kind == GRASS:
                # Variants 0-2 are tufts and flowers, anything higher is plain grass
                pattern = GRASS_TILES[min(variant, 3)]
            elif kind == FOREST:
                pattern = TREE
            elif kind == ROCK:
                pattern = ROCK_TILE
            elif kind == CAVE:
                pattern = CAVE_TILE
            elif kind == RIVER:
                pattern = shift(WATER, frame)
            elif kind == DEEP_WATER:
                pattern = shift(DEEP_TILE, frame)
            elif kind == CLIFF:
                pattern = CLIFF_TILE
            elif kind == FARMLAND:
                pattern = FARM_TILE
            else:  # POND
                pattern = shift(POND_TILE, frame) if frame else POND_TILE
            self.cache[key] = self._build(pattern, palette)
        return self.cache[key]

    def draw_berries(self, screen, px, py, count, poison=False):
        size = self.scale
        colour = POISON_COLOUR if poison else BERRY_COLOUR
        for row, col in BERRY_SPOTS[:count]:
            pygame.draw.rect(screen, colour,
                             (px + col * size, py + row * size, size, size))


# One colour per terrain type for the minimap
MINIMAP_KEYS = {GRASS: "G", FOREST: "L", ROCK: "R", RIVER: "W", POND: "P",
                CAVE: "K", DEEP_WATER: "D", CLIFF: "C", FARMLAND: "f"}


def minimap_colour(kind, winter):
    palette = WINTER if winter else SUMMER
    return palette[MINIMAP_KEYS[kind]]


# ---------- agents ----------
# "." means see-through
AGENT = ["..hhhh..",
         "..hssh..",
         "..ssss..",
         ".tttttt.",
         "s.tttt.s",
         "..tttt..",
         "..p..p..",
         "..p..p.."]

BONES = ["........",
         "..wwww..",
         ".wwwwww.",
         ".wkwwkw.",
         ".wwwwww.",
         "..wkkw..",
         "..wwww..",
         "........"]

AGENT_PALETTES = {
    0: {"h": (60, 40, 28), "s": (224, 172, 128), "t": (52, 92, 180), "p": (90, 62, 40)},    # male
    1: {"h": (110, 56, 30), "s": (224, 172, 128), "t": (196, 64, 72), "p": (90, 62, 40)},  # female
}
BONES_PALETTE = {"w": (232, 226, 210), "k": (40, 34, 34)}


def build_sprite(pattern, palette, tile_px):
    """Like SpriteSheet._build but with see-through pixels."""
    small = pygame.Surface((PIXELS, PIXELS), pygame.SRCALPHA)
    for y, row in enumerate(pattern):
        for x, ch in enumerate(row):
            if ch != ".":
                small.set_at((x, y), palette[ch])
    return pygame.transform.scale(small, (tile_px, tile_px))


CHILD = ["........",
         "........",
         "...hh...",
         "...ss...",
         "..tttt..",
         "..stts..",
         "...tt...",
         "...pp..."]

PREGNANT = ["..hhhh..",
            "..hssh..",
            "..ssss..",
            ".tttttt.",
            "s.tttttt",
            "..ttttt.",
            "..p..p..",
            "..p..p.."]

HEART = ["........",
         ".rr.rr..",
         "rrrrrrr.",
         "rrrrrrr.",
         ".rrrrr..",
         "..rrr...",
         "...r....",
         "........"]


def agent_sprites(tile_px):
    """Returns ({(sex, kind): image}, bones, heart) for one zoom level,
    where kind is "adult", "child" or "pregnant"."""
    people = {}
    for sex, pal in AGENT_PALETTES.items():
        people[(sex, "adult")] = build_sprite(AGENT, pal, tile_px)
        people[(sex, "child")] = build_sprite(CHILD, pal, tile_px)
        people[(sex, "pregnant")] = build_sprite(PREGNANT, pal, tile_px)
    bones = build_sprite(BONES, BONES_PALETTE, tile_px)
    heart = build_sprite(HEART, {"r": (240, 80, 120)}, tile_px)
    return people, bones, heart


# ---------- animals ----------
RABBIT_ART = ["........",
              ".....e.e",
              ".....e.e",
              "...eeeee",
              ".weeeeke",
              "..eeeeee",
              "...e..e.",
              "........"]

DEER_ART = [".....b.b",
            "......b.",
            ".....ddk",
            ".dddddd.",
            "dddddddd",
            ".llllll.",
            ".d.d..d.",
            ".d.d..d."]

WOLF_ART = ["......g.",
            ".....ggg",
            "g....gkG",
            ".ggggggg",
            ".gggggg.",
            ".GGGGGG.",
            ".g.g..g.",
            ".g.g..g."]

BEAR_ART = [".BB..BB.",
            ".BBBBBB.",
            "BBkBBkBB",
            "BBBBBBBB",
            ".BBnnBB.",
            "BBBBBBBB",
            "BBBBBBBB",
            "BB.BB.BB"]

ANIMAL_PALETTE = {
    "e": (176, 150, 120), "w": (245, 245, 240), "k": (20, 18, 18),
    "d": (170, 110, 60), "b": (226, 206, 160), "l": (232, 210, 176),
    "g": (150, 150, 158), "G": (98, 98, 108),
    "B": (88, 58, 36), "n": (150, 110, 80),
}
ANIMAL_ART = [RABBIT_ART, DEER_ART, WOLF_ART, BEAR_ART]   # same order as species numbers
WOLF_EYE = (2, 6)   # (row, column) of the wolf's eye, glows at night


def animal_sprites(tile_px):
    """Returns a list of images, one per species, for one zoom level."""
    return [build_sprite(art, ANIMAL_PALETTE, tile_px) for art in ANIMAL_ART]


# ---------- disasters ----------
ASH = ["aaAaaaaa",
       "aaaaaAaa",
       "AaaaaaaE",
       "aaaEaaaa",
       "aaaaaaAa",
       "aAaaaaaa",
       "aaaaEaaa",
       "aaaaaaaA"]

FIRE_FRAMES = [["....y...",
                "...yo...",
                "..yooy..",
                "..oroo.y",
                ".yorrroy",
                ".orrrrro",
                "orrRRrro",
                ".oRRRRo."],
               ["...y....",
                "...oy...",
                "..yooy..",
                "y.oorr..",
                "yorrroy.",
                "orrrrro.",
                "orrRRrro",
                ".oRRRRo."]]

FIRE_PALETTE = {"y": (255, 236, 120), "o": (255, 160, 40), "r": (232, 72, 30), "R": (150, 30, 20)}
ASH_PALETTE = {"a": (64, 60, 58), "A": (92, 86, 80), "E": (200, 90, 40)}


def disaster_sprites(tile_px):
    """Ash tile, fire animation frames and a flood overlay for one zoom level."""
    ash = build_sprite(ASH, ASH_PALETTE, tile_px)
    fire = [build_sprite(f, FIRE_PALETTE, tile_px) for f in FIRE_FRAMES]
    flood = pygame.Surface((tile_px, tile_px), pygame.SRCALPHA)
    flood.fill((60, 120, 200, 150))
    return ash, fire, flood


# ---------- farming, piles, weapons ----------
CROP_STAGES = [
    ["........",   # young sprouts
     "........",
     ".s..s..s",
     "........",
     "........",
     ".s..s..s",
     "........",
     "........"],
    ["........",   # growing
     ".s..s..s",
     ".s..s..s",
     "........",
     ".s..s..s",
     ".s..s..s",
     "........",
     "........"],
    [".y..y..y",   # ripe grain
     ".y..y..y",
     ".s..s..s",
     ".y..y..y",
     ".y..y..y",
     ".s..s..s",
     "........",
     "........"],
]
CROP_PALETTE = {"s": (96, 176, 64), "y": (240, 204, 84)}

PILE_ART = ["........",
            "........",
            "........",
            "...bb...",
            "..bmmb..",
            ".bmmmmb.",
            "..bbbb..",
            "........"]
PILE_PALETTE = {"b": (120, 84, 50), "m": (196, 146, 96)}

# Weapons drawn in the agent's right hand
WEAPON_ART = {
    1: ["........", "........", ".......b", ".......b", ".......b", ".......b", "........", "........"],
    2: ["........", "........", "........", "........", ".......g", "......gg", "........", "........"],
    3: [".......g", ".......b", ".......b", ".......b", ".......b", ".......b", ".......b", "........"],
}
WEAPON_PALETTE = {"b": (150, 100, 50), "g": (200, 200, 210)}


def farm_sprites(tile_px):
    """Crop overlays (young, growing, ripe), a pile image, and weapon overlays."""
    crops = [build_sprite(art, CROP_PALETTE, tile_px) for art in CROP_STAGES]
    pile = build_sprite(PILE_ART, PILE_PALETTE, tile_px)
    weapons = {k: build_sprite(art, WEAPON_PALETTE, tile_px) for k, art in WEAPON_ART.items()}
    return crops, pile, weapons
