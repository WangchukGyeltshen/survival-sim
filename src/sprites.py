"""Pixel art for everything in the game, drawn in code (no image files needed).

Every tile and sprite is a 16 x 16 grid of letters, where each letter is a
colour from a palette ("." means see-through). Some tiles (grass, water, rock)
are generated with a little randomness so neighbouring tiles don't look
identical; the rest are drawn by hand below. The grids are then scaled up so
each art pixel becomes a chunky square on screen.
"""

import random

import pygame

from src.world import (GRASS, FOREST, ROCK, RIVER, POND, CAVE,
                       DEEP_WATER, CLIFF, FARMLAND)

PIXELS = 16   # art pixels per tile side
OUTLINE = (16, 12, 18)

# ======================================================================
# colours
# ======================================================================
SUMMER = {
    "G": (104, 168, 76), "g": (138, 198, 96), "d": (82, 140, 60),            # grass
    "Y": (246, 214, 84), "M": (240, 240, 236), "Q": (226, 110, 150),         # flowers
    "L": (46, 112, 56), "l": (80, 154, 72), "D": (30, 80, 44),               # leaves
    "b": (112, 76, 44), "B": (82, 54, 30),                                    # trunk
    "R": (128, 128, 138), "r": (168, 168, 178), "k": (90, 90, 102),          # rock
    "K": (26, 20, 24), "c": (88, 74, 70), "C": (58, 46, 46),                 # cave, cliff
    "W": (58, 128, 212), "x": (140, 194, 242), "X": (44, 104, 184),          # river
    "O": (30, 66, 140), "o": (60, 104, 180),                                  # deep water
    "P": (52, 98, 122), "p": (92, 142, 162),                                  # pond
    "n": (70, 140, 64), "N": (50, 110, 50), "q": (240, 150, 180),            # lily pads
    "f": (128, 90, 56), "F": (96, 64, 40), "h": (150, 110, 72),              # farmland
    "a": (64, 60, 58), "A": (96, 90, 84), "E": (220, 100, 40),               # ash
}
WINTER = dict(SUMMER)
WINTER.update({
    "G": (222, 230, 238), "g": (242, 246, 250), "d": (190, 204, 220),
    "Y": (200, 212, 226), "M": (242, 246, 250), "Q": (200, 212, 226),
    "L": (44, 92, 70), "l": (236, 242, 248), "D": (28, 64, 50),
    "R": (140, 140, 150), "r": (232, 238, 244),
    "W": (96, 150, 205), "x": (200, 224, 244), "X": (80, 130, 190),
    "O": (60, 92, 150), "o": (100, 136, 190),
    "P": (110, 140, 160), "p": (150, 176, 192), "n": (110, 140, 160), "N": (100, 128, 148),
    "q": (110, 140, 160),
    "f": (200, 196, 196), "F": (170, 166, 170), "h": (222, 220, 222),
})

BERRY_COLOUR = (214, 40, 64)
BERRY_SHINE = (255, 150, 160)
POISON_COLOUR = (180, 80, 255)    # poisonous berries are purple
POISON_SHINE = (230, 190, 255)


def grid(fill="."):
    return [[fill] * PIXELS for _ in range(PIXELS)]


def rows(g):
    return ["".join(r) for r in g]


def check(art, name):
    assert len(art) == PIXELS and all(len(r) == PIXELS for r in art), f"{name} is not 16x16"
    return art


# ======================================================================
# generated terrain tiles
# ======================================================================
def grass_art(variant):
    rng = random.Random(1000 + variant)
    g = grid("G")
    for _ in range(14):
        g[rng.randrange(16)][rng.randrange(16)] = "d"
    for _ in range(10):
        g[rng.randrange(16)][rng.randrange(16)] = "g"
    for _ in range(3 if variant % 3 else 2):               # little tufts of grass
        x, y = rng.randrange(1, 15), rng.randrange(2, 15)
        g[y][x] = "g"
        g[y - 1][x] = "g"
        g[y][x - 1] = "d"
        g[y][x + 1] = "d"
    if variant in (2, 6):                                  # a flower
        x, y = rng.randrange(2, 14), rng.randrange(2, 14)
        petal = "Y" if variant == 2 else "Q"
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            g[y + dy][x + dx] = petal
        g[y][x] = "M" if variant == 2 else "Y"
    return rows(g)


def water_art(frame, base, light, dark, waves):
    rng = random.Random(7 if base == "W" else 9)
    g = grid(base)
    for _ in range(waves):
        y = rng.randrange(1, 15)
        x = (rng.randrange(16) + frame * 2) % 16
        length = rng.randrange(2, 5)
        for k in range(length):
            g[y][(x + k) % 16] = light
            g[y + 1][(x + k + 1) % 16] = dark
    return rows(g)


def pond_art(frame):
    g = [list(r) for r in water_art(frame, "P", "p", "P", 4)]
    for x, y in ((3, 4), (10, 9)):                         # lily pads
        for dx, dy in ((0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)):
            g[y + dy][x + dx] = "n"
        g[y][x + 1] = "N"
    g[9][11] = "q"
    return rows(g)


def rock_art(variant):
    rng = random.Random(500 + variant)
    g = grid("R")
    for _ in range(18):
        g[rng.randrange(16)][rng.randrange(16)] = "k"
    # a few boulders: light top-left edge, dark bottom-right edge
    for _ in range(3):
        cx, cy, r = rng.randrange(3, 13), rng.randrange(3, 13), rng.randrange(2, 4)
        for y in range(cy - r, cy + r + 1):
            for x in range(cx - r, cx + r + 1):
                if 0 <= x < 16 and 0 <= y < 16:
                    d = (x - cx) ** 2 + (y - cy) ** 2
                    if d <= r * r:
                        if d >= (r - 1) ** 2:
                            g[y][x] = "r" if (x <= cx and y <= cy) else "k"
                        else:
                            g[y][x] = "R"
    return rows(g)


def cave_art(variant):
    g = [list(r) for r in rock_art(variant)]
    for y in range(5, 16):
        for x in range(2, 14):
            # an arch: round top, straight sides
            if (x - 7.5) ** 2 / 30 + (y - 9) ** 2 / 18 <= 1 or (y >= 9 and 3 <= x <= 12):
                g[y][x] = "K"
    for y in range(5, 16):           # rim around the opening
        for x in range(1, 15):
            if g[y][x] != "K" and any(0 <= x + dx < 16 and 0 <= y + dy < 16 and g[y + dy][x + dx] == "K"
                                      for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                g[y][x] = "k"
    return rows(g)


def cliff_art(variant):
    rng = random.Random(800 + variant)
    g = [list(r) for r in rock_art(variant)]
    for x in range(16):
        g[4][x] = "r"                 # the edge
        g[5][x] = "k"
        for y in range(6, 16):
            g[y][x] = "c"
    for _ in range(5):               # cracks running down the face
        x = rng.randrange(16)
        for y in range(6, 16):
            if rng.random() < 0.85:
                g[y][x] = "C"
            x = max(0, min(15, x + rng.choice((-1, 0, 0, 1))))
    return rows(g)


def farm_art():
    g = grid("f")
    for y in (3, 7, 11, 15):
        for x in range(16):
            g[y][x] = "F"
            g[y - 1][x] = "h" if x % 5 else "f"
    return rows(g)


def ash_art():
    rng = random.Random(3)
    g = grid("a")
    for _ in range(30):
        g[rng.randrange(16)][rng.randrange(16)] = "A"
    for _ in range(4):
        g[rng.randrange(16)][rng.randrange(16)] = "E"
    return rows(g)


# ======================================================================
# hand-drawn art
# ======================================================================
TREE = check([
    ".....LLLLLL.....",
    "...LLlllLLLLL...",
    "..LlllLLLLLLLL..",
    ".LLllLLLLLLLLDL.",
    ".LlLLLLLLLLLLDD.",
    "LLLLLLLLLLLLLLDD",
    "LLLLLLLLLLLLLDDD",
    "LLLLLLLLLLLLDDDD",
    ".LLLLLLLLLLDDDD.",
    ".DLLLLLLLLDDDDD.",
    "..DDLLLLDDDDDD..",
    "...DDDDDDDDDD...",
    "......bbbB......",
    "......bbbB......",
    "......bbBB......",
    ".....bbbbBB.....",
], "tree")
BERRY_SPOTS = [(3, 6), (5, 10), (7, 3), (7, 8), (9, 5)]   # (row, col) on the tree

# ---------- people ----------
MAN = check([
    "................",
    "......hhhh......",
    ".....hhhhhh.....",
    ".....hssssh.....",
    ".....sesses.....",
    "......ssss......",
    ".......ss.......",
    ".....tttttt.....",
    "....tttttttt....",
    "...stuttttuts...",
    "...s.tttttt.s...",
    ".....vvvvvv.....",
    ".....pp..pp.....",
    ".....pp..pp.....",
    ".....pp..pp.....",
    ".....ff..ff.....",
], "man")
WOMAN = check([
    "................",
    "......hhhh......",
    ".....hhhhhh.....",
    "....hhsssshh....",
    "....hsessesh....",
    "....hhsssshh....",
    ".....h.ss.h.....",
    ".....tttttt.....",
    "....tttttttt....",
    "...stuttttuts...",
    "...s.tttttt.s...",
    "....tttttttt....",
    "...tttttttttt...",
    "...uuuuuuuuuu...",
    "......s..s......",
    ".....ff..ff.....",
], "woman")
PREGNANT = check([
    "................",
    "......hhhh......",
    ".....hhhhhh.....",
    "....hhsssshh....",
    "....hsessesh....",
    "....hhsssshh....",
    ".....h.ss.h.....",
    ".....tttttt.....",
    "....tttttttt....",
    "...stuttttutt...",
    "...s.ttttttttt..",
    "....tttttttttt..",
    "...tttttttttt...",
    "...uuuuuuuuuu...",
    "......s..s......",
    ".....ff..ff.....",
], "pregnant")
CHILD = check([
    "................",
    "................",
    "................",
    "................",
    "................",
    "................",
    "......hhhh......",
    "......hssh......",
    "......sees......",
    ".......ss.......",
    "......tttt......",
    ".....stttts.....",
    "......tttt......",
    "......p..p......",
    "......p..p......",
    "......f..f......",
], "child")
BONES = check([
    "................",
    "................",
    "................",
    ".....wwwwww.....",
    "....wwwwwwww....",
    "....wkkwwkkw....",
    "....wkkwwkkw....",
    "....wwwwwwww....",
    ".....wwkkww.....",
    "......wwww......",
    "...ww......ww...",
    ".....ww..ww.....",
    ".......ww.......",
    ".....ww..ww.....",
    "...ww......ww...",
    "................",
], "bones")
HEART = check([
    "................",
    "................",
    "....rrr..rrr....",
    "...rRrrrrrrrr...",
    "...rrrrrrrrrr...",
    "....rrrrrrrr....",
    ".....rrrrrr.....",
    "......rrrr......",
    ".......rr.......",
    "................",
    "................",
    "................",
    "................",
    "................",
    "................",
    "................",
], "heart")

SKIN_TONES = [(244, 204, 166), (222, 168, 122), (176, 122, 84), (122, 82, 56)]
HAIR_COLOURS = [(36, 28, 24), (84, 54, 32), (150, 76, 36), (214, 178, 100), (120, 120, 126)]
# Each tribe wears its own colour (tunic colour and its shade). Men and women
# are told apart by shape: men wear tunics and trousers, women long hair and dresses.
TRIBE_NAMES = ["Blue", "Red", "Green", "Gold"]
TRIBE_CLOTHES = [((40, 110, 245), (26, 76, 190)),
                 ((230, 50, 70), (176, 30, 50)),
                 ((60, 170, 80), (38, 124, 56)),
                 ((236, 186, 40), (190, 140, 20))]
TRIBE_COLOURS = [c for c, _ in TRIBE_CLOTHES]
PEOPLE_FIXED = {"e": (30, 24, 24), "p": (70, 52, 40), "f": (44, 32, 24), "v": (110, 76, 40)}

# ---------- animals ----------
RABBIT_ART = check([
    "................",
    "................",
    "................",
    "................",
    "..........ee....",
    "..........eq....",
    ".........eeq....",
    ".........ee.....",
    "........eeee....",
    ".......eeeeke...",
    "...weeeeeeeen...",
    "..wweeeeeeee....",
    "...eeeeeeeee....",
    "...eeE...eeE....",
    "................",
    "................",
], "rabbit")
DEER_ART = check([
    "..........a..a..",
    "...........aa...",
    "...........dd...",
    "...........ddk..",
    "..........ddddn.",
    "..........dd....",
    "..dddddddddd....",
    ".ddddddddddd....",
    "dddddddddddd....",
    ".dllllllllld....",
    "..d.d....d.d....",
    "..d.d....d.d....",
    "..d.d....d.d....",
    "..H.H....H.H....",
    "................",
    "................",
], "deer")
WOLF_ART = check([
    "................",
    "................",
    "................",
    "...........g.g..",
    "..........gggg..",
    "..........ggkgG.",
    "g.........gggGG.",
    "gg..gggggggggg..",
    ".ggggggggggggg..",
    "..gggggggggggg..",
    "..GGGGGGGGGGG...",
    "..g.g.....g.g...",
    "..g.g.....g.g...",
    "..G.G.....G.G...",
    "................",
    "................",
], "wolf")
BEAR_ART = check([
    "................",
    "................",
    "................",
    "...........BB...",
    "..BBBBBBBBBBBBB.",
    ".BBBBBBBBBBBkBB.",
    "BBBBBBBBBBBBBnn.",
    "BBBBBBBBBBBBBB..",
    "BBBBBBBBBBBBBB..",
    "BuuuuuuuuuuuBB..",
    ".BB.BB...BB.BB..",
    ".BB.BB...BB.BB..",
    ".BB.BB...BB.BB..",
    ".KK.KK...KK.KK..",
    "................",
    "................",
], "bear")
ANIMAL_PALETTE = {
    "e": (176, 150, 120), "q": (232, 160, 170), "k": (20, 18, 18), "n": (60, 40, 40),
    "w": (245, 245, 240), "E": (130, 108, 86),
    "d": (176, 112, 60), "l": (236, 214, 180), "a": (226, 206, 160), "H": (60, 40, 30),
    "g": (150, 150, 158), "G": (98, 98, 108),
    "B": (88, 58, 36), "u": (124, 88, 58), "K": (40, 30, 24),
}
ANIMAL_ART = [RABBIT_ART, DEER_ART, WOLF_ART, BEAR_ART]   # same order as species numbers
WOLF_EYE = (5, 12)   # (row, column) of the wolf's eye, glows at night

# ---------- items ----------
PILE_ART = check([
    "................",
    "................",
    "................",
    "................",
    "................",
    "................",
    "................",
    "................",
    "..........n.....",
    "......nnn.n.....",
    ".....nmmmnn.....",
    "....nmmMmmmn....",
    "...nmmmmmmmmn...",
    "....nnnnnnnn....",
    "................",
    "................",
], "pile")
PILE_PALETTE = {"n": (120, 84, 50), "m": (200, 110, 100), "M": (240, 170, 160)}


def weapon_art(kind):
    g = grid()
    if kind == 1:      # stick
        for y in range(3, 14):
            g[y][13] = "n" if y % 3 else "N"
    elif kind == 2:    # sharp stone in the hand
        for y, x in ((9, 13), (10, 12), (10, 13), (10, 14), (11, 13), (11, 14)):
            g[y][x] = "g"
        g[9][13] = "G"
    elif kind == 4:    # flint blade: a dark, sharp leaf shape
        for y, x in ((7, 13), (8, 13), (8, 14), (9, 13), (9, 14), (10, 13), (11, 13)):
            g[y][x] = "f"
        g[7][13] = "F"
    else:              # spears: long shaft with a stone or metal tip
        tip = {3: "g", 5: "c", 6: "b", 7: "i"}[kind]
        for y in range(3, 15):
            g[y][13] = "n" if y % 3 else "N"
        for y, x in ((0, 13), (1, 12), (1, 13), (1, 14), (2, 13)):
            g[y][x] = tip
        g[0][13] = "G"
    return rows(g)


WEAPON_PALETTE = {"n": (150, 100, 50), "N": (110, 72, 36), "g": (200, 200, 210), "G": (250, 250, 255),
                  "f": (70, 74, 96), "F": (150, 160, 190), "c": (214, 120, 60), "b": (226, 180, 70),
                  "i": (170, 180, 196)}


def crop_art(stage):
    g = grid()
    for c in (2, 6, 10, 14):
        for base in (2, 6, 10, 14):           # just above each furrow
            if stage == 0:                    # young sprouts
                g[base][c] = "s"
                g[base - 1][c - 1] = "s"
                g[base - 1][c + 1 if c < 15 else c] = "s"
            elif stage == 1:                  # growing stalks
                g[base][c] = "S"
                g[base - 1][c] = "s"
                g[base - 2][c - 1] = "s"
                g[base - 2][c + 1 if c < 15 else c] = "s"
            else:                             # ripe golden grain
                g[base][c] = "S"
                g[base - 1][c] = "y"
                g[base - 2][c] = "Y"
                g[base - 2][c - 1] = "y"
                g[base - 2][c + 1 if c < 15 else c] = "y"
    return rows(g)


CROP_PALETTE = {"s": (110, 190, 72), "S": (70, 140, 50), "y": (244, 208, 90), "Y": (255, 236, 150)}


def fire_art(frame):
    rng = random.Random(40 + frame)
    g = grid()
    for x in range(1, 15):
        height = 5 + int(6 * (1 - abs(x - 7.5) / 8)) + rng.randrange(0, 4)
        for k in range(height):
            y = 15 - k
            if k < height * 0.35:
                g[y][x] = "R"
            elif k < height * 0.65:
                g[y][x] = "r"
            elif k < height * 0.9:
                g[y][x] = "o"
            else:
                g[y][x] = "y"
    return rows(g)


FIRE_PALETTE = {"y": (255, 236, 120), "o": (255, 160, 40), "r": (232, 72, 30), "R": (150, 30, 20)}

# Small icons shown above an agent's head when it does something (5 x 5)
ICONS = {
    "drink": ["..b..", ".bbb.", "bbbbb", "bbbbb", ".bbb."],
    "eat": ["...g.", ".rrr.", "rrrrr", "rrrrr", ".rrr."],
    "attack": ["x...x", ".x.x.", "..x..", ".x.x.", "x...x"],
    "pick up": ["..w..", ".www.", "w.w.w", "..w..", "..w.."],
    "plant": ["g...g", ".g.g.", "..g..", "..g..", ".nnn."],
    "craft": ["kkkk.", "kkkk.", "..n..", "..n..", "..n.."],
    "mate": [".r.r.", "rrrrr", "rrrrr", ".rrr.", "..r.."],
    "give": ["..r..", ".rrr.", "..r..", "nnnnn", ".nnn."],
    "build": ["kkkkk", "k.k.k", "kkkkk", "k.k.k", "kkkkk"],
    "call 1": ["wwww.", "w..w.", "wwww.", ".w...", "....."],
    "call 2": ["wwwww", "w.w.w", "wwwww", ".w...", "....."],
    "call 3": ["wwwww", "wcwcw", "wwwww", ".w...", "....."],
    "sick": [".ggg.", "g.g.g", "ggggg", "g.g.g", ".ggg."],
}
ICON_PALETTE = {"b": (80, 170, 255), "r": (240, 60, 90), "g": (90, 200, 80), "x": (255, 70, 50),
                "w": (250, 250, 250), "n": (150, 100, 50), "k": (180, 180, 190), "c": (255, 200, 60)}


# ======================================================================
# turning letter grids into images
# ======================================================================
def to_surface(art, palette, outline=False):
    """Small (16 x 16, or 18 x 18 with outline) surface from a letter grid."""
    n = len(art)
    pad = 1 if outline else 0
    surf = pygame.Surface((n + 2 * pad, n + 2 * pad), pygame.SRCALPHA)
    filled = set()
    for y, row in enumerate(art):
        for x, ch in enumerate(row):
            if ch != ".":
                filled.add((x + pad, y + pad))
                surf.set_at((x + pad, y + pad), palette[ch])
    if outline:
        for x, y in filled:
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if (x + dx, y + dy) not in filled:
                    surf.set_at((x + dx, y + dy), OUTLINE)
    return surf


def scaled(surf, tile_px, art_size=PIXELS):
    """Scale so that art_size art pixels = tile_px screen pixels (hard pixel edges)."""
    w, h = surf.get_size()
    return pygame.transform.scale(surf, (round(w * tile_px / art_size), round(h * tile_px / art_size)))


def build_sprite(art, palette, tile_px):
    return scaled(to_surface(art, palette), tile_px)


def build_outlined(art, palette, tile_px):
    """A sprite with a 1-pixel dark outline, so it stands out from the ground.
    It is one art pixel bigger on every side, so draw it at outline_offset()."""
    return scaled(to_surface(art, palette, outline=True), tile_px)


def outline_offset(tile_px):
    return round(tile_px / PIXELS)


def overlay(base, top):
    """Lay one letter grid over another ('.' in the top grid lets the base show)."""
    return ["".join(t if t != "." else b for b, t in zip(br, tr)) for br, tr in zip(base, top)]


class SpriteSheet:
    """Terrain tiles for one zoom level, built when first needed and then cached."""

    def __init__(self, tile_px):
        self.tile_px = tile_px
        self.cache = {}

    def get(self, kind, variant, winter, frame):
        key = (kind, variant, winter, frame)
        if key not in self.cache:
            palette = WINTER if winter else SUMMER
            v = int(variant) % 10
            if kind == GRASS:
                art = grass_art(v)
            elif kind == FOREST:
                art = overlay(grass_art(v), TREE)
            elif kind == ROCK:
                art = rock_art(v % 4)
            elif kind == CAVE:
                art = cave_art(v % 2)
            elif kind == CLIFF:
                art = cliff_art(v % 3)
            elif kind == RIVER:
                art = water_art(frame, "W", "x", "X", 6)
            elif kind == DEEP_WATER:
                art = water_art(frame, "O", "o", "O", 4)
            elif kind == FARMLAND:
                art = farm_art()
            else:  # POND
                art = pond_art(frame % 2)
            self.cache[key] = build_sprite(art, palette, self.tile_px)
        return self.cache[key]

    def draw_berries(self, screen, px, py, count, poison=False):
        s = self.tile_px / PIXELS
        colour, shine = (POISON_COLOUR, POISON_SHINE) if poison else (BERRY_COLOUR, BERRY_SHINE)
        size = max(1, round(2 * s))
        for row, col in BERRY_SPOTS[:count]:
            x, y = px + round(col * s), py + round(row * s)
            pygame.draw.rect(screen, colour, (x, y, size, size))
            if s >= 2:
                pygame.draw.rect(screen, shine, (x, y, round(s), round(s)))


# One colour per terrain type for the minimap
MINIMAP_KEYS = {GRASS: "G", FOREST: "L", ROCK: "R", RIVER: "W", POND: "P",
                CAVE: "K", DEEP_WATER: "O", CLIFF: "c", FARMLAND: "f"}


def minimap_colour(kind, winter):
    palette = WINTER if winter else SUMMER
    return palette[MINIMAP_KEYS[kind]]


def people_palette(tribe, skin, hair):
    tunic, shade = TRIBE_CLOTHES[tribe]
    pal = dict(PEOPLE_FIXED)
    pal.update({"s": SKIN_TONES[skin], "h": HAIR_COLOURS[hair], "t": tunic, "u": shade})
    return pal


class PeopleSprites(dict):
    """People images, made the first time each look is needed:
    key = (sex, kind, skin, hair, tribe), kind = "adult", "child" or "pregnant"."""

    def __init__(self, tile_px):
        super().__init__()
        self.tile_px = tile_px

    def __missing__(self, key):
        sex, kind, skin, hair, tribe = key
        art = CHILD if kind == "child" else (PREGNANT if kind == "pregnant" else (MAN if sex == 0 else WOMAN))
        image = build_outlined(art, people_palette(tribe, skin, hair), self.tile_px)
        self[key] = image
        return image


def agent_sprites(tile_px):
    """Returns (people images, bones, heart) for one zoom level."""
    people = PeopleSprites(tile_px)
    bones = build_outlined(BONES, {"w": (232, 226, 210), "k": (40, 34, 34)}, tile_px)
    heart = build_outlined(HEART, {"r": (240, 80, 120), "R": (255, 190, 210)}, tile_px)
    return people, bones, heart


def animal_sprites(tile_px):
    """A list of images, one per species, for one zoom level."""
    return [build_outlined(art, ANIMAL_PALETTE, tile_px) for art in ANIMAL_ART]


def disaster_sprites(tile_px):
    """Ash tile, fire animation frames and a flood overlay for one zoom level."""
    ash = build_sprite(ash_art(), SUMMER, tile_px)
    fire = [build_sprite(fire_art(f), FIRE_PALETTE, tile_px) for f in range(2)]
    flood = pygame.Surface((tile_px, tile_px), pygame.SRCALPHA)
    flood.fill((60, 120, 200, 150))
    s = max(1, tile_px // PIXELS)
    for y in (4, 11):
        pygame.draw.line(flood, (170, 210, 250, 170), (2 * s, y * s), (7 * s, y * s), s)
    return ash, fire, flood


def farm_sprites(tile_px):
    """Crop overlays (young, growing, ripe), a pile image, and weapon overlays."""
    crops = [build_sprite(crop_art(stage), CROP_PALETTE, tile_px) for stage in range(3)]
    pile = build_outlined(PILE_ART, PILE_PALETTE, tile_px)
    weapons = {k: build_outlined(weapon_art(k), WEAPON_PALETTE, tile_px) for k in range(1, 8)}
    return crops, pile, weapons


def shadow_sprite(tile_px):
    """A soft dark oval drawn under feet."""
    s = tile_px / PIXELS
    surf = pygame.Surface((tile_px, tile_px), pygame.SRCALPHA)
    pygame.draw.ellipse(surf, (0, 0, 0, 80), (round(3 * s), round(13 * s), round(10 * s), round(3 * s)))
    return surf


def icon_sprites(tile_px):
    scale = max(1, round(tile_px / 12))
    out = {}
    for name, art in ICONS.items():
        small = to_surface(art, ICON_PALETTE, outline=True)
        out[name] = pygame.transform.scale(small, (7 * scale, 7 * scale))
    return out


# ======================================================================
# Stage 2: resource deposits and structures
# ======================================================================
def deposit_art(kind):
    """Small marks drawn on top of a tile: flint and clay on the ground,
    ores as coloured specks in the rock."""
    rng = random.Random(70 + kind)
    g = grid()
    if kind == 1:                                 # flint: a few grey-blue nodules
        for _ in range(3):
            x, y = rng.randrange(2, 13), rng.randrange(3, 13)
            for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
                g[y + dy][x + dx] = "f"
            g[y][x] = "F"
    elif kind == 2:                               # clay: reddish lumps
        for _ in range(2):
            x, y = rng.randrange(2, 11), rng.randrange(3, 12)
            for dy in range(3):
                for dx in range(4 - (dy == 2)):
                    g[y + dy][x + dx] = "c"
            g[y][x + 1] = "C"
    else:                                         # ore specks in the rock
        mark = {3: "o", 4: "t", 5: "i"}[kind]
        for _ in range(9 if kind != 4 else 6):
            x, y = rng.randrange(1, 15), rng.randrange(1, 15)
            g[y][x] = mark
            if rng.random() < 0.5:
                g[y][min(x + 1, 15)] = mark.upper()
    return rows(g)


DEPOSIT_PALETTE = {"f": (110, 116, 140), "F": (200, 206, 224), "c": (178, 98, 64), "C": (214, 140, 100),
                   "o": (60, 180, 140), "O": (140, 230, 190), "t": (210, 214, 224), "T": (250, 250, 255),
                   "i": (170, 70, 44), "I": (220, 120, 80)}

CAMPFIRE_ART = [check([
    "................",
    "................",
    "................",
    "................",
    "................",
    ".......y........",
    "......yoy.......",
    "......oro.y.....",
    ".....yorro......",
    ".....orrro......",
    "....oorRroo.....",
    "...bbbbbbbbb....",
    "..bBbbBbbBbbb...",
    "...bbbbbbbbb....",
    "..kk.......kk...",
    "................",
], "campfire")]
CAMPFIRE_ART.append(rows([[("." if c in "yo" and i < 7 else c) for c in r] for i, r in
                          enumerate([list(r) for r in CAMPFIRE_ART[0]])]))
HUT_ART = check([
    "................",
    ".......YY.......",
    "......YyyY......",
    ".....YyyyyY.....",
    "....YyYyyYyY....",
    "...YyyyyyyyyY...",
    "..YyyYyyyYyyyY..",
    ".YYYYYYYYYYYYYY.",
    "..bwwwwwwwwwwb..",
    "..bwwwwwwwwwwb..",
    "..bwwwwKKwwwwb..",
    "..bwwwKKKKwwwb..",
    "..bwwwKKKKwwwb..",
    "..bwwwKKKKwwwb..",
    "..bbbbKKKKbbbb..",
    "................",
], "hut")
FENCE_ART = check([
    "................",
    "................",
    "................",
    "..b....b....b...",
    ".bBb..bBb..bBb..",
    "bbbbbbbbbbbbbbbb",
    "BBBBBBBBBBBBBBBB",
    ".bBb..bBb..bBb..",
    ".bBb..bBb..bBb..",
    "bbbbbbbbbbbbbbbb",
    "BBBBBBBBBBBBBBBB",
    ".bBb..bBb..bBb..",
    ".bBb..bBb..bBb..",
    ".bBb..bBb..bBb..",
    "................",
    "................",
], "fence")
WALL_ART = check([
    "................",
    "rrrrrrrrrrrrrrrr",
    "RRRkRRRRRkRRRRRk",
    "RRRkRRRRRkRRRRRk",
    "kkkkkkkkkkkkkkkk",
    "RkRRRRRkRRRRRRkR",
    "RkRRRRRkRRRRRRkR",
    "kkkkkkkkkkkkkkkk",
    "RRRRkRRRRRkRRRRR",
    "RRRRkRRRRRkRRRRR",
    "kkkkkkkkkkkkkkkk",
    "RRkRRRRRRkRRRRkR",
    "RRkRRRRRRkRRRRkR",
    "kkkkkkkkkkkkkkkk",
    "KKKKKKKKKKKKKKKK",
    "................",
], "wall")
STORAGE_ART = check([
    "................",
    "................",
    "................",
    "................",
    "...bbbbbbbbbb...",
    "..bBBBBBBBBBBb..",
    "..bbbbbbbbbbbb..",
    "..bnnnnnnnnnnb..",
    "..bnnnnyynnnnb..",
    "..bnnnnyynnnnb..",
    "..bnnnnnnnnnnb..",
    "..bnnnnnnnnnnb..",
    "..bbbbbbbbbbbb..",
    "...B........B...",
    "................",
    "................",
], "storage")
BRIDGE_ART = check([
    "................",
    "BBBBBBBBBBBBBBBB",
    "bbbbbbbbbbbbbbbb",
    "nnnnnnnnnnnnnnnn",
    "bBbbbBbbbbBbbbBb",
    "nnnnnnnnnnnnnnnn",
    "bbbBbbbbbBbbbbbb",
    "nnnnnnnnnnnnnnnn",
    "bbbbbbBbbbbbBbbb",
    "nnnnnnnnnnnnnnnn",
    "bBbbbbbbBbbbbbbB",
    "nnnnnnnnnnnnnnnn",
    "bbbbbbbbbbbbbbbb",
    "BBBBBBBBBBBBBBBB",
    "................",
    "................",
], "bridge")
WELL_ART = check([
    "................",
    "..bbbbbbbbbbbb..",
    "..b..........b..",
    "..b....kk....b..",
    "..b....k.....b..",
    "..b....k.....b..",
    "...rrrrrrrrrr...",
    "..rRRRRRRRRRRr..",
    "..RkWWWWWWWWkR..",
    "..RkWwWWWwWWkR..",
    "..RkWWWWWWWWkR..",
    "..RRRRRRRRRRRR..",
    "..RkRRkRRkRRkR..",
    "..kkkkkkkkkkkk..",
    "................",
    "................",
], "well")
TOWER_ART = check([
    "...YYYYYYYYYY...",
    "..YyyyyyyyyyyY..",
    "..bBBBBBBBBBBb..",
    "..b.b......b.b..",
    "..bbbbbbbbbbbb..",
    "...b.b....b.b...",
    "...b..b..b..b...",
    "...b...bb...b...",
    "...b...bb...b...",
    "...b..b..b..b...",
    "...b.b....b.b...",
    "...bb......bb...",
    "...b.b....b.b...",
    "...b..b..b..b...",
    "..RRR......RRR..",
    "..kkk......kkk..",
], "watchtower")
FURNACE_ART = check([
    "................",
    "......kkkk......",
    "......kRRk......",
    "......kRRk......",
    "....rrRRRRrr....",
    "...rRRRRRRRRr...",
    "..rRRRkkkkRRRr..",
    "..RRRkooookRRR..",
    "..RRkoyyyyokRR..",
    "..RRkoyrryokRR..",
    "..RRkoooooo kR..".replace(" ", "k"),
    "..RRRRRRRRRRRR..",
    "..RkRRkRRkRRkR..",
    "..kkkkkkkkkkkk..",
    "................",
    "................",
], "furnace")
STRUCTURE_PALETTE = {
    "y": (214, 186, 90), "Y": (170, 140, 60), "w": (196, 160, 110), "b": (130, 88, 48), "B": (92, 60, 32),
    "n": (164, 116, 66), "K": (40, 30, 26), "k": (84, 84, 96), "r": (182, 182, 192), "R": (138, 138, 150),
    "W": (58, 128, 212), "o": (255, 150, 40), "y2": (255, 236, 120), "r2": (232, 72, 30),
}
FLAME = {"y": (255, 236, 120), "o": (255, 160, 40), "r": (232, 72, 30), "R": (150, 30, 20),
         "b": (120, 80, 44), "B": (84, 56, 30), "k": (90, 90, 100)}


def structure_sprites(tile_px):
    """{structure number: image}; the campfire is a list of 2 flickering frames."""
    from src.world import CAMPFIRE, HUT, FENCE, WALL, STORAGE, BRIDGE, WELL, WATCHTOWER, FURNACE
    furnace_pal = dict(STRUCTURE_PALETTE)
    furnace_pal.update({"y": (255, 236, 120), "r": (232, 72, 30)})
    return {
        CAMPFIRE: [build_outlined(f, FLAME, tile_px) for f in CAMPFIRE_ART],
        HUT: build_outlined(HUT_ART, STRUCTURE_PALETTE, tile_px),
        FENCE: build_outlined(FENCE_ART, STRUCTURE_PALETTE, tile_px),
        WALL: build_sprite(WALL_ART, STRUCTURE_PALETTE, tile_px),
        STORAGE: build_outlined(STORAGE_ART, STRUCTURE_PALETTE, tile_px),
        BRIDGE: build_sprite(BRIDGE_ART, STRUCTURE_PALETTE, tile_px),
        WELL: build_outlined(WELL_ART, STRUCTURE_PALETTE, tile_px),
        WATCHTOWER: build_outlined(TOWER_ART, STRUCTURE_PALETTE, tile_px),
        FURNACE: build_outlined(FURNACE_ART, furnace_pal, tile_px),
    }


def deposit_sprites(tile_px):
    return {kind: build_sprite(deposit_art(kind), DEPOSIT_PALETTE, tile_px) for kind in range(1, 6)}
