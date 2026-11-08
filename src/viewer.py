"""Draws the world in a window using pygame, so it can be watched like a game.

Layout:  [ top bar ........................................ ]
         [ map (camera view)                 | sidebar:     ]
         [                                   |  population  ]
         [                                   |  event log   ]

Controls:
    W A S D / arrow keys   move the camera
    mouse drag             move the camera
    mouse wheel            zoom in / out (over the log: scroll the log)
    click the minimap      jump to that spot
    click an agent         select it and see its stats
    click a log line       select that agent and follow it
    F                      camera follows the selected agent
    V                      show what the selected agent sees
    I                      show / hide action icons
    + / -                  speed up / slow down
    SPACE                  pause / resume
    M                      show / hide minimap
    H                      show / hide help
    R                      new random world
    ESC                    quit
"""

import os
import random
from collections import deque

import numpy as np
import pygame

from src.world import POND, FARMLAND, CROP_GROW_TICKS, TICKS_PER_DAY, DAYLIGHT_TICKS
from src.sprites import (SpriteSheet, PIXELS, minimap_colour, agent_sprites, animal_sprites,
                         WOLF_EYE, disaster_sprites, farm_sprites, shadow_sprite, icon_sprites,
                         outline_offset, SKIN_TONES, HAIR_COLOURS)
from src.agents import (ITEMS, WEAPON_NAMES, HANDS, GENE_NAMES, ADULT_AGE, EAT, DRINK,
                        PICK_UP, PLANT, CRAFT, ATTACK_ANIMAL, ATTACK_PERSON, MATE,
                        ACTION_NAMES, DEATH_CAUSES, TICKS_PER_YEAR, MALE)
from src.observe import observe, CH, VIEW_SIZE, TERRAIN_ORDER
from src.animals import RABBIT, DEER, WOLF, BEAR
from src.simulation import Simulation

# Window layout (the real sizes are worked out from the screen in setup_window)
HUD_HEIGHT = 40
SIDEBAR_W = 430
MAP_W, MAP_H = 1450, 940

ZOOMS = [8, 16, 32, 48]          # tile sizes in screen pixels (16 x 16 art)
START_ZOOM = 2                   # 32 pixel tiles
PAN_SPEED = 900                  # camera speed in screen pixels per second
MINIMAP_SCALE = 2                # minimap pixels per tile

SPEEDS = [1, 2, 5, 10, 20, 50, 100, 300]   # ticks per second
GRAVE_TICKS = TICKS_PER_DAY      # bones stay visible for one day
ICON_TICKS = 6                   # action icons stay up for this many ticks
HIT_TICKS = 3                    # red hit slashes stay this many ticks
HEART_TICKS = 12                 # hearts float up for this many ticks
ICON_FOR_ACTION = {EAT: "eat", DRINK: "drink", PICK_UP: "pick up", PLANT: "plant", CRAFT: "craft",
                   ATTACK_ANIMAL: "attack", ATTACK_PERSON: "attack", MATE: "mate"}
SELECT_COLOUR = (255, 230, 90)
BAR_COLOURS = {
    "health": (214, 60, 70),
    "food": (230, 150, 60),
    "water": (70, 150, 230),
    "energy": (240, 210, 80),
    "warmth": (240, 120, 160),
}

HUD_BG = (24, 20, 28)
PANEL_BG = (32, 28, 38)
BORDER = (64, 56, 74)
HUD_TEXT = (236, 228, 210)
MUTED = (150, 142, 160)
SUN = (250, 204, 70)
MOON = (200, 210, 235)
MAN_COLOUR = (110, 165, 255)
WOMAN_COLOUR = (255, 120, 140)
SEASON_COLOURS = {
    "spring": (140, 210, 110),
    "summer": (250, 204, 70),
    "autumn": (220, 130, 60),
    "winter": (170, 200, 240),
}
HELP_LINES = [
    "WASD / arrows   move camera",
    "mouse drag      move camera",
    "mouse wheel     zoom",
    "click minimap   jump there",
    "click agent     select it",
    "click log line  follow that agent",
    "F               follow selected",
    "V               agent's eyes",
    "I               action icons",
    "+ / -           speed",
    "SPACE           pause",
    "M               minimap",
    "R               new world",
    "T               turbo (training)",
    "H               close help",
    "ESC             quit",
]

# ---------- event log ----------
# Which filter each kind of event belongs to
CATEGORY = {
    "conceived": "family", "born": "family",
    "died": "deaths",
    "poisoned": "health", "got sick": "health",
    "attacked": "fights", "killed": "fights", "hunted": "fights", "hit animal": "fights",
    "crafted": "work", "planted": "work", "harvested": "work", "weapon broke": "work", "looted": "work",
    "disaster": "disasters", "discovery": "discoveries",
    "ate": "food", "drank": "food",
    "picked up": "items",
}
FILTERS = [   # (category, button label, on at start, colour)
    ("family", "Family", True, (255, 140, 180)),
    ("deaths", "Deaths", True, (230, 80, 80)),
    ("health", "Health", True, (160, 220, 100)),
    ("fights", "Fights", True, (255, 120, 60)),
    ("work", "Work", True, (200, 170, 110)),
    ("disasters", "Disasters", True, (255, 214, 120)),
    ("discoveries", "Discoveries", True, (140, 230, 120)),
    ("food", "Food", False, (120, 180, 255)),
    ("items", "Items", False, (190, 190, 200)),
]
DEATH_WORDS = {
    "starvation": "starved to death", "thirst": "died of thirst", "wolf": "was killed by a wolf",
    "bear": "was killed by a bear", "cold": "froze to death", "poison": "died from poison",
    "sickness": "died of sickness", "drowned": "drowned", "fell": "fell off a cliff",
    "flood": "was swept away by a flood", "fire": "burned in a fire",
    "lightning": "was struck by lightning", "old age": "died of old age",
    "deer": "was kicked to death by a deer", "childbirth": "died in childbirth",
}
SYLLABLES_A = ["ka", "mi", "ta", "lo", "ri", "sa", "no", "pe", "da", "yu", "ke", "bo", "ni", "ra",
               "zu", "ve", "ha", "jo", "li", "ma", "to", "se", "wa", "go"]
SYLLABLES_B = ["ra", "n", "vo", "la", "mi", "ko", "ta", "ri", "sh", "do", "na", "ki", "lu", "m",
               "re", "ya", "s", "to", "ba", "l"]


def night_darkness(tick):
    """0 = full daylight, 1 = darkest night. Fades in and out smoothly."""
    t = tick % TICKS_PER_DAY
    fade = 8  # ticks of dusk / dawn
    if t < DAYLIGHT_TICKS - fade:
        return 0.0
    if t < DAYLIGHT_TICKS:
        return (t - (DAYLIGHT_TICKS - fade)) / fade
    if t >= TICKS_PER_DAY - fade:
        return (TICKS_PER_DAY - t) / fade
    return 1.0


def setup_window():
    """Make the window as big as fits nicely on this screen."""
    global MAP_W, MAP_H
    try:   # on Windows, ask for real pixels so pixel art stays sharp
        import ctypes
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass
    os.environ.setdefault("SDL_VIDEO_CENTERED", "1")
    pygame.init()
    info = pygame.display.Info()
    screen_w = info.current_w if info.current_w and info.current_w > 0 else 1920
    screen_h = info.current_h if info.current_h and info.current_h > 0 else 1080
    width = max(1100, min(1880, screen_w - 40))
    height = max(700, min(1000, screen_h - 110))
    MAP_W = width - SIDEBAR_W
    MAP_H = height - HUD_HEIGHT
    return pygame.display.set_mode((width, height))


class Viewer:
    def __init__(self, sim):
        self.screen = setup_window()
        pygame.display.set_caption("Survival Sim")
        self.font = pygame.font.SysFont("consolas", 18, bold=True)
        self.small_font = pygame.font.SysFont("consolas", 15, bold=True)
        self.log_font = pygame.font.SysFont("consolas", 15)
        self.clock = pygame.time.Clock()
        self.sheets = {size: SpriteSheet(size) for size in ZOOMS}
        self.people = {size: agent_sprites(size) for size in ZOOMS}
        self.beasts = {size: animal_sprites(size) for size in ZOOMS}
        self.hazards = {size: disaster_sprites(size) for size in ZOOMS}
        self.farm = {size: farm_sprites(size) for size in ZOOMS}
        self.shadows = {size: shadow_sprite(size) for size in ZOOMS}
        self.icons = {size: icon_sprites(size) for size in ZOOMS}
        self.show_icons = True
        self.hit_marks = []      # (x, y, tick) red slashes where hits landed
        self.heart_marks = []    # (x, y, tick) hearts where a child was conceived
        self.show_eyes = False
        # Training mode (set by train.py)
        self.training = False
        self.turbo = False         # T: stop drawing so training runs at full speed
        self.training_info = []    # lines of text shown in the training panel
        self.new_world_maker = lambda: Simulation(size=self.world.size)
        self.flash = 0           # frames of lightning flash left
        self.rain = np.random.default_rng(1).random((260, 2))   # raindrop positions
        self.zoom = START_ZOOM
        self.speed_index = 3   # start at 10 ticks per second
        self.paused = False
        self.show_minimap = True
        self.show_help = False
        self.mouse_down_at = None
        self.dragging = False
        self.selected = None     # index of the selected agent
        self.following = False
        self.night = pygame.Surface((MAP_W, MAP_H), pygame.SRCALPHA)
        # Event log
        self.filters = {cat: on for cat, _, on, _ in FILTERS}
        self.selected_only = False
        self.log_scroll = 0           # lines scrolled back from the newest
        self.filter_buttons = []      # (rect, category) for clicks
        self.log_line_rects = []      # (rect, agent) for clicks
        self.set_sim(sim)

    @property
    def tile(self):
        return ZOOMS[self.zoom]

    def set_sim(self, sim):
        self.sim = sim
        self.world = world = sim.world
        self.selected = None
        self.following = False
        self.hit_marks = []      # a new world starts with a clean slate
        self.heart_marks = []
        # Pick a random look for each tile once, so the map doesn't flicker
        self.variants = np.random.default_rng(0).integers(0, 10, (world.size, world.size))
        self.minimap = None
        self.minimap_season = None
        self.looks = {}               # agent slot -> (spawn tick, name, skin, hair)
        self.log_lines = deque(maxlen=4000)
        self.log_seen = sim.log_count
        self.log_scroll = 0
        # Start with the camera in the middle of the world
        self.cam_x = world.size * self.tile / 2 - MAP_W / 2
        self.cam_y = world.size * self.tile / 2 - MAP_H / 2
        self.clamp_camera()

    # ---------- names and looks (only for display) ----------
    def look(self, i):
        """(name, skin, hair) of agent i. Made up the first time we need it;
        children take their mother's skin and one parent's hair."""
        a = self.sim.agents
        i = int(i)
        spawn = int(a.spawn_tick[i])
        known = self.looks.get(i)
        if known and known[0] == spawn:
            return known[1:]
        rnd = random.Random(i * 7919 + spawn * 104729)
        name = (rnd.choice(SYLLABLES_A) + rnd.choice(SYLLABLES_B) +
                (rnd.choice(SYLLABLES_A) if rnd.random() < 0.4 else "")).capitalize()
        skin, hair = rnd.randrange(len(SKIN_TONES)), rnd.randrange(len(HAIR_COLOURS))
        mother, father = int(a.mother[i]), int(a.father[i])
        if mother >= 0 and mother in self.looks:
            skin = self.looks[mother][2]
            parents = [p for p in (mother, father) if p >= 0 and p in self.looks]
            hair = self.looks[rnd.choice(parents)][3]
        self.looks[i] = (spawn, name, skin, hair)
        return name, skin, hair

    def name(self, i):
        return self.look(i)[0]

    # ---------- camera ----------
    def clamp_camera(self):
        """Stop the camera from moving past the edges of the world."""
        world_px = self.world.size * self.tile
        self.cam_x = min(max(self.cam_x, 0), max(world_px - MAP_W, 0))
        self.cam_y = min(max(self.cam_y, 0), max(world_px - MAP_H, 0))

    def set_zoom(self, new_zoom, focus):
        """Zoom while keeping the point under the mouse in the same place."""
        new_zoom = min(max(new_zoom, 0), len(ZOOMS) - 1)
        if new_zoom == self.zoom:
            return
        fx, fy = focus
        world_x = (self.cam_x + fx) / self.tile   # tile under the mouse
        world_y = (self.cam_y + fy) / self.tile
        self.zoom = new_zoom
        self.cam_x = world_x * self.tile - fx
        self.cam_y = world_y * self.tile - fy
        self.clamp_camera()

    def center_on(self, tile_x, tile_y):
        self.cam_x = tile_x * self.tile - MAP_W / 2
        self.cam_y = tile_y * self.tile - MAP_H / 2
        self.clamp_camera()

    def to_screen(self, tile_x, tile_y):
        return (tile_x * self.tile - int(self.cam_x),
                HUD_HEIGHT + tile_y * self.tile - int(self.cam_y))

    def agent_at(self, screen_x, screen_y):
        """Index of a living agent on the clicked tile, or None."""
        tx = int((self.cam_x + screen_x) // self.tile)
        ty = int((self.cam_y + screen_y - HUD_HEIGHT) // self.tile)
        a = self.sim.agents
        found = np.flatnonzero(a.alive & (a.x == tx) & (a.y == ty))
        return int(found[0]) if len(found) else None

    # ---------- drawing the map ----------
    def draw_map(self):
        w = self.world
        tile = self.tile
        sheet = self.sheets[tile]
        winter = w.season == "winter"
        frame = (pygame.time.get_ticks() // 500) % 2   # water ripples every half second

        # Only draw the tiles the camera can see
        first_x, first_y = int(self.cam_x // tile), int(self.cam_y // tile)
        last_x = min(first_x + MAP_W // tile + 2, w.size)
        last_y = min(first_y + MAP_H // tile + 2, w.size)

        self.screen.set_clip((0, HUD_HEIGHT, MAP_W, MAP_H))
        self.screen.fill((10, 10, 14), (0, HUD_HEIGHT, MAP_W, MAP_H))
        ash, fire_frames, flood = self.hazards[tile]
        crops, pile_img, _ = self.farm[tile]
        off = outline_offset(tile)
        jobs, overlays, fires, piles = [], [], [], []
        berry_tiles = []
        for y in range(first_y, last_y):
            py = HUD_HEIGHT + y * tile - int(self.cam_y)
            for x in range(first_x, last_x):
                kind = w.terrain[y, x]
                px = x * tile - int(self.cam_x)
                if w.burnt[y, x] > 0:
                    jobs.append((ash, (px, py)))
                else:
                    jobs.append((sheet.get(kind, self.variants[y, x], winter, frame), (px, py)))
                if kind == FARMLAND:
                    if w.berries[y, x]:
                        overlays.append((crops[2], (px, py)))          # ripe
                    else:
                        stage = 0 if w.crop[y, x] > CROP_GROW_TICKS // 2 else 1
                        overlays.append((crops[stage], (px, py)))
                elif w.berries[y, x]:
                    berry_tiles.append((px, py, int(w.berries[y, x]), bool(w.poison[y, x])))
                if (x, y) in w.piles:
                    piles.append((pile_img, (px - off, py - off)))
                if w.flood[y, x] > 0:
                    overlays.append((flood, (px, py)))
                if w.fire[y, x] > 0:
                    fires.append((fire_frames[(frame + x + y) % 2], (px, py)))
        self.screen.blits(jobs, doreturn=False)
        for px, py, count, poison in berry_tiles:
            sheet.draw_berries(self.screen, px, py, count, poison)
        self.screen.blits(overlays, doreturn=False)
        self.screen.blits(piles, doreturn=False)
        self.draw_agents(first_x, first_y, last_x, last_y)

        # Night: dark blue see-through layer over the map
        darkness = night_darkness(w.tick)
        alpha = int(160 * darkness)
        if alpha:
            self.night.fill((12, 14, 48, alpha))
            self.screen.blit(self.night, (0, HUD_HEIGHT))
            if darkness > 0.5:
                self.draw_wolf_eyes(first_x, first_y, last_x, last_y)
        # Fire is drawn after the night layer so it glows in the dark
        self.screen.blits(fires, doreturn=False)
        self.draw_weather()
        self.screen.set_clip(None)

    def draw_weather(self):
        w = self.world
        if self.sim.disasters.lightning:
            self.flash = 4
        if w.storm > 0:
            self.night.fill((40, 44, 56, 70))
            self.screen.blit(self.night, (0, HUD_HEIGHT))
            # Falling rain (or snow in winter), moved along over time
            t = pygame.time.get_ticks() / 1000
            snow = w.season == "winter"
            speed = 60 if snow else 500
            for fx, fy in self.rain:
                x = int((fx * MAP_W + t * (20 if snow else -120)) % MAP_W)
                y = HUD_HEIGHT + int((fy * MAP_H + t * speed) % MAP_H)
                if snow:
                    pygame.draw.rect(self.screen, (240, 244, 250), (x, y, 3, 3))
                else:
                    pygame.draw.line(self.screen, (170, 190, 230), (x, y), (x - 3, y + 10), 1)
        if self.flash > 0:
            self.night.fill((255, 255, 255, 30 * self.flash))
            self.screen.blit(self.night, (0, HUD_HEIGHT))
            self.flash -= 1

    def draw_wolf_eyes(self, first_x, first_y, last_x, last_y):
        """Wolves' eyes glow in the dark, drawn on top of the night layer."""
        an = self.sim.animals
        s = self.tile / PIXELS
        size = max(1, round(s))
        row, col = WOLF_EYE
        wolves = an.alive & (an.species == WOLF) & (an.x >= first_x) & (an.x < last_x) \
            & (an.y >= first_y) & (an.y < last_y)
        for i in np.flatnonzero(wolves):
            px, py = self.to_screen(an.x[i], an.y[i])
            pygame.draw.rect(self.screen, (255, 220, 60), (px + round(col * s), py + round(row * s), size, size))

    def draw_agents(self, first_x, first_y, last_x, last_y):
        a = self.sim.agents
        tile = self.tile
        people, bones, heart = self.people[tile]
        off = outline_offset(tile)   # outlined sprites are drawn one art-pixel up and left
        in_view = (a.x >= first_x) & (a.x < last_x) & (a.y >= first_y) & (a.y < last_y)

        # Bones where agents died recently, fading away
        recent = ~a.alive & in_view & (a.death_tick >= 0) & \
            (self.world.tick - a.death_tick < GRAVE_TICKS)
        for i in np.flatnonzero(recent):
            age = self.world.tick - a.death_tick[i]
            bones.set_alpha(int(255 * (1 - age / GRAVE_TICKS)))
            px, py = self.to_screen(a.x[i], a.y[i])
            self.screen.blit(bones, (px - off, py - off))
        bones.set_alpha(255)

        an = self.sim.animals
        beasts = self.beasts[tile]
        animals_in_view = an.alive & (an.x >= first_x) & (an.x < last_x) & \
            (an.y >= first_y) & (an.y < last_y)
        shadow = self.shadows[tile]
        for i in np.flatnonzero(animals_in_view):
            px, py = self.to_screen(an.x[i], an.y[i])
            self.screen.blit(shadow, (px, py))
            self.screen.blit(beasts[an.species[i]], (px - off, py - off))

        weapons = self.farm[tile][2]
        adult_age = ADULT_AGE * TICKS_PER_YEAR
        icons = self.icons[tile]
        living = np.flatnonzero(a.alive & in_view)
        for i in living:
            px, py = self.to_screen(a.x[i], a.y[i])
            kind = "child" if a.age[i] < adult_age else ("pregnant" if a.pregnant[i] else "adult")
            _, skin, hair = self.look(i)
            self.screen.blit(shadow, (px, py))
            self.screen.blit(people[(int(a.sex[i]), kind, skin, hair)], (px - off, py - off))
            if a.weapon[i] != HANDS:
                self.screen.blit(weapons[a.weapon[i]], (px - off, py - off))
        # Icons above heads for actions that just worked (drawn last so nothing covers them)
        if self.show_icons:
            for i in living:
                if self.world.tick - a.did_tick[i] < ICON_TICKS and a.did[i] in ICON_FOR_ACTION:
                    px, py = self.to_screen(a.x[i], a.y[i])
                    icon = icons[ICON_FOR_ACTION[a.did[i]]]
                    self.screen.blit(icon, (px + (tile - icon.get_width()) // 2, py - icon.get_height() + 2))

        # Red slashes where hits landed (they last a few ticks of game time)
        now = self.world.tick
        width = max(2, tile // 10)
        for x, y, t in self.hit_marks:
            px, py = self.to_screen(x, y)
            pygame.draw.line(self.screen, (240, 40, 40), (px + 3, py + 3), (px + tile - 4, py + tile - 4), width)
            pygame.draw.line(self.screen, (255, 200, 200), (px + 5, py + 3), (px + tile - 2, py + tile - 6), 1)
        for x, y, t in self.heart_marks:
            px, py = self.to_screen(x, y)
            rise = (now - t) * tile // (2 * HEART_TICKS)   # floats up
            self.screen.blit(heart, (px - off, py - tile // 2 - rise))

        # Selected agent: yellow corner brackets and its name
        if self.selected is not None and a.alive[self.selected]:
            px, py = self.to_screen(a.x[self.selected], a.y[self.selected])
            k = max(4, tile // 4)
            for cx, cy, dx, dy in ((px - 2, py - 2, 1, 1), (px + tile + 1, py - 2, -1, 1),
                                   (px - 2, py + tile + 1, 1, -1), (px + tile + 1, py + tile + 1, -1, -1)):
                pygame.draw.line(self.screen, SELECT_COLOUR, (cx, cy), (cx + dx * k, cy), 2)
                pygame.draw.line(self.screen, SELECT_COLOUR, (cx, cy), (cx, cy + dy * k), 2)
            label = self.small_font.render(self.name(self.selected), False, SELECT_COLOUR)
            box = label.get_rect(midtop=(px + tile // 2, py + tile + 4)).inflate(8, 2)
            pygame.draw.rect(self.screen, HUD_BG, box)
            self.screen.blit(label, label.get_rect(center=box.center))

    # ---------- panels on the map ----------
    def panel(self, rect, alpha=225, border=BORDER):
        surf = pygame.Surface(rect.size, pygame.SRCALPHA)
        surf.fill((*HUD_BG, alpha))
        self.screen.blit(surf, rect)
        pygame.draw.rect(self.screen, border, rect, 2)

    def draw_events(self):
        """Recent disaster messages (yellow) and first discoveries (green),
        top-left of the map."""
        tick = self.world.tick
        events = [(t, m, (255, 214, 120)) for t, m in self.sim.disasters.events[-4:]]
        events += [(t, f"Day {t // TICKS_PER_DAY}: {m}", (140, 230, 120))
                   for t, m in self.sim.discoveries[-4:]]
        events.sort(key=lambda e: e[0])
        recent = [e for e in events if tick - e[0] < 300][-4:]
        for row, (t, message, colour) in enumerate(recent):
            label = self.small_font.render(message, False, colour)
            box = label.get_rect(topleft=(12, HUD_HEIGHT + 12 + row * 26)).inflate(14, 8)
            self.panel(box, 200, (50, 44, 58))
            self.screen.blit(label, (12, HUD_HEIGHT + 12 + row * 26))

    def draw_agent_panel(self):
        """Stats of the selected agent, bottom-left of the map."""
        i = self.selected
        if i is None:
            return
        a = self.sim.agents
        height = 292 if a.alive[i] else 64
        box = pygame.Rect(12, HUD_HEIGHT + MAP_H - height - 12, 480, height)
        self.panel(box)

        sex = "male" if a.sex[i] == MALE else "female"
        colour = MAN_COLOUR if a.sex[i] == MALE else WOMAN_COLOUR
        name = self.font.render(self.name(i), False, colour)
        self.screen.blit(name, (box.x + 12, box.y + 10))
        years = a.age[i] / TICKS_PER_YEAR
        info = self.small_font.render(f"{sex}, {years:.1f} years, generation {a.generation[i]}", False, MUTED)
        self.screen.blit(info, (box.x + 20 + name.get_width(), box.y + 12))

        if not a.alive[i]:
            cause = DEATH_CAUSES[a.cause_of_death[i]]
            label = self.small_font.render(f"DEAD ({cause})", False, (214, 60, 70))
            self.screen.blit(label, (box.x + 12, box.y + 38))
            return
        tags = []
        if a.sick[i] > 0:
            tags.append(("SICK", (150, 220, 90)))
        if a.pregnant[i] > 0:
            tags.append((f"PREGNANT {a.pregnant[i] / TICKS_PER_DAY:.1f} days", (240, 120, 160)))
        if self.following:
            tags.append(("FOLLOWING", SELECT_COLOUR))
        tx = box.x + 12
        for text, tag_colour in tags:
            tag = self.small_font.render(text, False, tag_colour)
            self.screen.blit(tag, (tx, box.y + 36))
            tx += tag.get_width() + 12

        y = box.y + 60
        for stat, bar_colour in BAR_COLOURS.items():
            value = getattr(a, stat)[i]
            self.screen.blit(self.small_font.render(stat, False, HUD_TEXT), (box.x + 12, y))
            pygame.draw.rect(self.screen, (50, 44, 56), (box.x + 90, y + 3, 200, 12))
            pygame.draw.rect(self.screen, bar_colour, (box.x + 90, y + 3, int(2 * value), 12))
            self.screen.blit(self.small_font.render(f"{value:.0f}", False, MUTED), (box.x + 300, y))
            y += 21
        lines = [f"doing: {ACTION_NAMES[a.last_action[i]]}"]
        weapon = WEAPON_NAMES[a.weapon[i]]
        if a.weapon[i] != HANDS:
            weapon += f" ({a.weapon_uses[i]} uses left)"
        lines.append(f"weapon: {weapon}")
        carried = ", ".join(f"{a.inv[item][i]} {item}" for item in ITEMS if a.inv[item][i])
        lines.append(f"carrying: {carried or 'nothing'}")
        family = f"children: {a.children[i]}   people killed: {a.kills[i]}"
        mother = int(a.mother[i])
        if mother >= 0 and int(a.spawn_tick[i]) >= int(a.spawn_tick[mother]):
            family = f"mother: {self.name(mother)}   " + family
        lines.append(family)
        for k, line in enumerate(lines):
            self.screen.blit(self.small_font.render(line, False, HUD_TEXT), (box.x + 12, y + 4 + k * 21))
        short = {"strength": "str", "hardiness": "hardy", "endurance": "endur", "vision": "vision"}
        genes = "genes: " + "  ".join(f"{short[g]} {a.genes[g][i]:.2f}" for g in GENE_NAMES)
        self.screen.blit(self.small_font.render(genes, False, (180, 200, 240)), (box.x + 12, y + 4 + 4 * 21))

    def build_minimap(self):
        w = self.world
        winter = w.season == "winter"
        colours = np.zeros((w.size, w.size, 3), dtype=np.uint8)
        for kind in np.unique(w.terrain):
            colours[w.terrain == kind] = minimap_colour(kind, winter)
        # pygame expects (x, y) order, so swap the axes
        surface = pygame.surfarray.make_surface(colours.swapaxes(0, 1))
        size = w.size * MINIMAP_SCALE
        self.minimap = pygame.transform.scale(surface, (size, size))
        self.minimap_season = w.season

    def minimap_rect(self):
        size = self.world.size * MINIMAP_SCALE
        return pygame.Rect(MAP_W - size - 14, HUD_HEIGHT + MAP_H - size - 14, size, size)

    def draw_minimap(self):
        if self.minimap is None or self.minimap_season != self.world.season:
            self.build_minimap()
        rect = self.minimap_rect()
        pygame.draw.rect(self.screen, HUD_BG, rect.inflate(8, 8))
        pygame.draw.rect(self.screen, BORDER, rect.inflate(8, 8), 2)
        self.screen.blit(self.minimap, rect)
        a = self.sim.agents
        for i in np.flatnonzero(a.alive):
            colour = MAN_COLOUR if a.sex[i] == MALE else WOMAN_COLOUR
            self.screen.fill(colour, (rect.x + int(a.x[i] * MINIMAP_SCALE), rect.y + int(a.y[i] * MINIMAP_SCALE), 2, 2))
        an = self.sim.animals
        for i in np.flatnonzero(an.alive & np.isin(an.species, (WOLF, BEAR))):
            self.screen.fill((230, 40, 40), (rect.x + int(an.x[i] * MINIMAP_SCALE), rect.y + int(an.y[i] * MINIMAP_SCALE), 2, 2))
        # White box showing what the camera sees
        scale = MINIMAP_SCALE / self.tile
        view = pygame.Rect(rect.x + self.cam_x * scale, rect.y + self.cam_y * scale,
                           MAP_W * scale, MAP_H * scale).clip(rect)
        pygame.draw.rect(self.screen, (255, 255, 255), view, 1)

    # Plain colours for the debug view. The brain itself gets no colours at all,
    # just "channel number 14 is on"; these colours only help us humans read it.
    EYE_TERRAIN = [(90, 150, 70), (40, 100, 50), (120, 120, 128), (60, 130, 210),
                   (70, 110, 140), (40, 30, 30), (30, 60, 140), (80, 60, 60), (130, 90, 55)]
    EYE_MARKS = [("red berries", (220, 50, 60)), ("purple berries", (190, 90, 255)),
                 ("crop ripe", (240, 205, 80)), ("crop growing", (110, 190, 80)),
                 ("pile", (200, 150, 100)), ("ash", (70, 66, 64)), ("flood", (90, 150, 230)),
                 ("fire", (255, 140, 40)), ("rabbit", (200, 180, 150)), ("deer", (180, 120, 60)),
                 ("wolf", (160, 160, 170)), ("bear", (100, 66, 40)), ("child", (255, 255, 255)),
                 ("woman", (240, 100, 120)), ("man", (90, 130, 230))]

    def draw_eyes(self):
        """Debug panel: the selected agent's view, exactly as the brain receives it."""
        i = self.selected
        if not self.show_eyes or i is None or not self.sim.agents.alive[i]:
            return
        view, _ = observe(self.sim, [i])
        v = view[0]
        cell = 12
        size = VIEW_SIZE * cell
        box = pygame.Rect(MAP_W - size - 30, HUD_HEIGHT + 200, size + 16, size + 40)
        self.panel(box, 240)
        label = self.small_font.render(f"What {self.name(i)} sees", False, HUD_TEXT)
        self.screen.blit(label, (box.x + 8, box.y + 8))
        ox, oy = box.x + 8, box.y + 32
        for ty in range(VIEW_SIZE):
            for tx in range(VIEW_SIZE):
                rect = (ox + tx * cell, oy + ty * cell, cell, cell)
                if not v[CH["visible"], ty, tx]:
                    pygame.draw.rect(self.screen, (12, 10, 16), rect)   # can't see
                    continue
                colour = (0, 0, 0)
                for k in range(len(TERRAIN_ORDER)):
                    if v[k, ty, tx]:
                        colour = self.EYE_TERRAIN[k]
                pygame.draw.rect(self.screen, colour, rect)
                for mark_name, mark in self.EYE_MARKS:
                    if v[CH[mark_name], ty, tx] > 0:
                        pygame.draw.rect(self.screen, mark, (rect[0] + 2, rect[1] + 2, cell - 4, cell - 4))
                        break
        c = VIEW_SIZE // 2
        pygame.draw.rect(self.screen, (255, 230, 90), (ox + c * cell, oy + c * cell, cell, cell), 1)

    def draw_help(self):
        line_h = 21
        box = pygame.Rect(16, HUD_HEIGHT + 120, 340, len(HELP_LINES) * line_h + 20)
        self.panel(box, 235)
        for k, line in enumerate(HELP_LINES):
            label = self.small_font.render(line, False, HUD_TEXT)
            self.screen.blit(label, (box.x + 12, box.y + 10 + k * line_h))

    def draw_training(self):
        """Training progress panel (only while training), top-right of the map."""
        if not self.training_info:
            return
        line_h = 21
        width = 340
        box = pygame.Rect(MAP_W - width - 14, HUD_HEIGHT + 12, width, len(self.training_info) * line_h + 14)
        self.panel(box, 225, (110, 84, 170))
        for row, line in enumerate(self.training_info):
            colour = (200, 170, 255) if row == 0 else HUD_TEXT
            label = self.small_font.render(line, False, colour)
            self.screen.blit(label, (box.x + 10, box.y + 7 + row * line_h))

    # ---------- top bar ----------
    def text(self, string, x, colour, y=10):
        """Draw text without smoothing (keeps the pixel look). Returns the end x."""
        label = self.font.render(string, False, colour)
        self.screen.blit(label, (x, y))
        return x + label.get_width()

    def draw_hud(self):
        w = self.world
        width = self.screen.get_width()
        pygame.draw.rect(self.screen, HUD_BG, (0, 0, width, HUD_HEIGHT))
        pygame.draw.rect(self.screen, BORDER, (0, HUD_HEIGHT - 3, width, 3))

        # Sun or moon icon (pixel squares)
        if w.is_night:
            pygame.draw.rect(self.screen, MOON, (12, 11, 16, 16))
            pygame.draw.rect(self.screen, HUD_BG, (19, 8, 13, 13))
        else:
            pygame.draw.rect(self.screen, SUN, (12, 12, 16, 16))
            for dx, dy in ((-6, 6), (20, 6), (6, -6), (6, 20)):
                pygame.draw.rect(self.screen, SUN, (12 + dx, 12 + dy, 4, 4))

        hour = (w.tick % TICKS_PER_DAY) * 24 // TICKS_PER_DAY
        x = 44
        x = self.text(f"DAY {w.day}", x, HUD_TEXT)
        x = self.text(f"{hour:02d}:00", x + 12, MUTED)
        x = self.text(w.season.upper(), x + 20, SEASON_COLOURS[w.season])
        x = self.text(f"YEAR {w.day // 20 + 1}", x + 14, MUTED)

        # Berry counter with a little berry icon
        pygame.draw.rect(self.screen, (214, 40, 64), (x + 22, 15, 10, 10))
        pygame.draw.rect(self.screen, (78, 150, 74), (x + 25, 10, 4, 5))
        x = self.text(f"{int(w.berries.sum()):,}", x + 38, HUD_TEXT)

        a = self.sim.agents
        x = self.text(f"ALIVE {a.population}", x + 26, HUD_TEXT)
        dead = sum(self.sim.deaths.values())
        x = self.text(f"DEAD {dead}", x + 18, (214, 90, 90))
        x = self.text(f"BORN {a.births}", x + 18, (240, 120, 160))
        top_gen = int(a.generation[a.alive].max()) if a.population else 0
        x = self.text(f"GEN {top_gen}", x + 18, (180, 200, 240))

        # Active disasters
        if w.storm > 0:
            x = self.text("BLIZZARD" if w.season == "winter" else "STORM", x + 24, (170, 190, 230))
        if (w.flood > 0).any():
            x = self.text("FLOOD", x + 16, (90, 150, 230))
        if (w.fire > 0).any():
            x = self.text("FIRE", x + 16, (255, 140, 40))

        speed = "MAX" if self.training and self.speed_index == len(SPEEDS) - 1 else f"x{SPEEDS[self.speed_index]}"
        status = "PAUSED" if self.paused else speed
        label = self.font.render(f"{status}   H: help", False, HUD_TEXT)
        self.screen.blit(label, (width - label.get_width() - 14, 10))

    # ---------- sidebar: population and event log ----------
    def draw_sidebar(self):
        sx = MAP_W
        h = self.screen.get_height()
        pygame.draw.rect(self.screen, PANEL_BG, (sx, HUD_HEIGHT, SIDEBAR_W, h - HUD_HEIGHT))
        pygame.draw.rect(self.screen, BORDER, (sx, HUD_HEIGHT, 3, h - HUD_HEIGHT))
        x0 = sx + 16
        y = HUD_HEIGHT + 12

        # --- population ---
        self.screen.blit(self.small_font.render("POPULATION", False, MUTED), (x0, y))
        y += 24
        a = self.sim.agents
        adult = a.age >= ADULT_AGE * TICKS_PER_YEAR
        men = int((a.alive & adult & (a.sex == MALE)).sum())
        women = int((a.alive & adult & (a.sex != MALE)).sum())
        kids = int((a.alive & ~adult).sum())
        people, _, _ = self.people[ZOOMS[2]]
        off = outline_offset(ZOOMS[2])
        col_w = (SIDEBAR_W - 32) // 3
        for k, (key, count, label) in enumerate((((0, "adult", 1, 1), men, "men"),
                                                 ((1, "adult", 0, 2), women, "women"),
                                                 ((1, "child", 1, 0), kids, "children"))):
            cx = x0 + k * col_w
            self.screen.blit(people[key], (cx - off, y - off))
            self.screen.blit(self.font.render(str(count), False, HUD_TEXT), (cx + 38, y + 1))
            self.screen.blit(self.log_font.render(label, False, MUTED), (cx + 38, y + 20))
        y += 46
        beasts = self.beasts[ZOOMS[2]]
        an = self.sim.animals
        col_w = (SIDEBAR_W - 32) // 4
        for k, (kind, label) in enumerate(((RABBIT, "rabbits"), (DEER, "deer"), (WOLF, "wolves"), (BEAR, "bears"))):
            cx = x0 + k * col_w
            self.screen.blit(beasts[kind], (cx - off, y - off))
            self.screen.blit(self.font.render(str(an.count(kind)), False, HUD_TEXT), (cx + 36, y + 1))
            self.screen.blit(self.log_font.render(label, False, MUTED), (cx + 36, y + 20))
        y += 50
        pygame.draw.line(self.screen, BORDER, (x0, y), (sx + SIDEBAR_W - 16, y), 2)
        y += 12

        # --- filter buttons ---
        self.screen.blit(self.small_font.render("EVENT LOG", False, MUTED), (x0, y))
        hint = self.log_font.render("click a line to follow", False, (110, 104, 120))
        self.screen.blit(hint, (sx + SIDEBAR_W - hint.get_width() - 16, y))
        y += 26
        self.filter_buttons = []
        bx = x0
        buttons = [(cat, label, colour) for cat, label, _, colour in FILTERS]
        buttons.append(("selected only", "Selected only", SELECT_COLOUR))
        for cat, label, colour in buttons:
            on = self.selected_only if cat == "selected only" else self.filters[cat]
            text = self.log_font.render(label, True, (20, 18, 24) if on else MUTED)
            rect = pygame.Rect(bx, y, text.get_width() + 18, 24)
            if rect.right > sx + SIDEBAR_W - 12:
                bx = x0
                y += 30
                rect.topleft = (bx, y)
            pygame.draw.rect(self.screen, colour if on else (46, 40, 54), rect, border_radius=12)
            if not on:
                pygame.draw.rect(self.screen, colour, rect, 1, border_radius=12)
            self.screen.blit(text, (rect.x + 9, rect.y + 4))
            self.filter_buttons.append((rect, cat))
            bx = rect.right + 6
        y += 36

        # --- log lines (newest at the top) ---
        visible = [line for line in reversed(self.log_lines) if self.shown(line)]
        line_h = 21
        rows = (h - y - 10) // line_h
        self.log_scroll = max(0, min(self.log_scroll, max(len(visible) - rows, 0)))
        if self.log_scroll:
            note = self.log_font.render(f"^ {self.log_scroll} newer (scroll up)", False, SELECT_COLOUR)
            self.screen.blit(note, (x0, y - 18))
        self.log_line_rects = []
        for tick, cat, segments, who in visible[self.log_scroll:self.log_scroll + rows]:
            day = tick // TICKS_PER_DAY
            hour = (tick % TICKS_PER_DAY) * 24 // TICKS_PER_DAY
            stamp = self.log_font.render(f"D{day} {hour:02d}h", True, (110, 104, 120))
            self.screen.blit(stamp, (x0, y))
            dot = dict((c, col) for c, _, _, col in FILTERS).get(cat, MUTED)
            pygame.draw.circle(self.screen, dot, (x0 + 86, y + 9), 4)
            tx = x0 + 96
            for text, colour in segments:
                label = self.log_font.render(text, True, colour)
                if tx + label.get_width() > sx + SIDEBAR_W - 8:
                    break
                self.screen.blit(label, (tx, y))
                tx += label.get_width()
            if who:
                self.log_line_rects.append((pygame.Rect(sx, y, SIDEBAR_W, line_h), who[0]))
            y += line_h
        if not visible:
            self.screen.blit(self.log_font.render("nothing yet...", True, MUTED), (x0, y))

    def shown(self, line):
        tick, cat, segments, who = line
        if not self.filters.get(cat, True):
            return False
        if self.selected_only:
            return self.selected is not None and self.selected in who
        return True

    def person(self, i):
        """A coloured name segment for the log."""
        colour = MAN_COLOUR if self.sim.agents.sex[i] == MALE else WOMAN_COLOUR
        return (self.name(i), colour)

    def describe(self, kind, a, b):
        """Turn one event into coloured text pieces, and the agents involved."""
        text = HUD_TEXT
        if kind == "disaster":
            return [(b, (255, 214, 120))], []
        if kind == "discovery":
            return [(b, (140, 230, 120))], []
        who = [a]
        if kind == "died":
            if b == "killed by person":
                return None, []          # the "killed" event already says who did it
            pieces = [self.person(a), (" " + DEATH_WORDS.get(b, f"died ({b})"), (230, 120, 120))]
        elif kind == "killed":
            pieces = [self.person(a), (" killed ", (255, 120, 90)), self.person(b)]
            who = [a, b]
        elif kind == "attacked":
            pieces = [self.person(a), (" attacked ", text), self.person(b)]
            who = [a, b]
        elif kind == "conceived":
            pieces = [self.person(a), (" and ", text), self.person(b), (" are expecting", text)]
            who = [a, b]
        elif kind == "born":
            pieces = [self.person(a), (" was born to ", text), self.person(b)]
            who = [a, b]
        elif kind == "hunted":
            pieces = [self.person(a), (f" hunted a {b}", (255, 180, 110))]
        elif kind == "hit animal":
            pieces = [self.person(a), (f" hit a {b}", text)]
        elif kind == "poisoned":
            pieces = [self.person(a), (" ate poisonous berries", (190, 120, 255))]
        elif kind == "got sick":
            pieces = [self.person(a), (" got sick from pond water", (160, 220, 100))]
        elif kind == "crafted":
            pieces = [self.person(a), (f" crafted {b}", (230, 200, 140))]
        elif kind == "planted":
            pieces = [self.person(a), (" planted a seed", (140, 220, 110))]
        elif kind == "harvested":
            pieces = [self.person(a), (" harvested a crop", (240, 210, 100))]
        elif kind == "weapon broke":
            pieces = [self.person(a), (f"'s {b} broke", text)]
        elif kind == "looted":
            pieces = [self.person(a), (" took things from a pile", text)]
        elif kind == "ate":
            pieces = [self.person(a), (f" ate {b}", text)]
        elif kind == "drank":
            pieces = [self.person(a), (f" drank {b}", text)]
        elif kind == "picked up":
            pieces = [self.person(a), (f" picked up {b}", text)]
        else:
            pieces = [self.person(a), (f" {kind} {b if b is not None else ''}", text)]
        return pieces, who

    def collect_log(self):
        """Turn this tick's events into log lines (names are fixed at this moment)."""
        count = self.sim.log_count - self.log_seen
        self.log_seen = self.sim.log_count
        if count <= 0:
            return
        log = self.sim.log
        count = min(count, len(log))
        new = [log[k] for k in range(len(log) - count, len(log))]
        for t, kind, a, b in new:
            if kind == "born":
                self.look(a)   # give the baby its family look while the mother is known
            pieces, who = self.describe(kind, a, b)
            if pieces:
                self.log_lines.append((t, CATEGORY.get(kind, "work"), pieces, who))
                if self.log_scroll and self.shown(self.log_lines[-1]):
                    self.log_scroll += 1   # keep the view still while scrolled back

    # ---------- controls ----------
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                if event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    self.speed_index = min(self.speed_index + 1, len(SPEEDS) - 1)
                if event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self.speed_index = max(self.speed_index - 1, 0)
                if event.key == pygame.K_m:
                    self.show_minimap = not self.show_minimap
                if event.key == pygame.K_h:
                    self.show_help = not self.show_help
                if event.key == pygame.K_r and not self.training:
                    self.set_sim(self.new_world_maker())
                if event.key == pygame.K_t and self.training:
                    self.turbo = not self.turbo
                if event.key == pygame.K_v:
                    self.show_eyes = not self.show_eyes
                if event.key == pygame.K_i:
                    self.show_icons = not self.show_icons
                if event.key == pygame.K_f and self.selected is not None:
                    self.following = not self.following

            if event.type == pygame.MOUSEWHEEL:
                mx, my = pygame.mouse.get_pos()
                if mx >= MAP_W:
                    self.log_scroll = max(0, self.log_scroll - event.y * 3)   # scroll the log
                else:
                    self.set_zoom(self.zoom + event.y, (mx, my - HUD_HEIGHT))

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if event.pos[0] >= MAP_W:
                    self.click_sidebar(event.pos)
                    continue
                rect = self.minimap_rect()
                if self.show_minimap and rect.collidepoint(event.pos):
                    tx = (event.pos[0] - rect.x) / MINIMAP_SCALE
                    ty = (event.pos[1] - rect.y) / MINIMAP_SCALE
                    self.center_on(tx, ty)
                elif event.pos[1] > HUD_HEIGHT:
                    self.mouse_down_at = event.pos
            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                if self.mouse_down_at and not self.dragging:
                    # A click (not a drag): select the agent under the mouse
                    self.selected = self.agent_at(*event.pos)
                    self.following = False
                self.mouse_down_at = None
                self.dragging = False
            if event.type == pygame.MOUSEMOTION and self.mouse_down_at:
                mx, my = self.mouse_down_at
                if abs(event.pos[0] - mx) + abs(event.pos[1] - my) > 5:
                    self.dragging = True
                if self.dragging:
                    self.cam_x -= event.rel[0]
                    self.cam_y -= event.rel[1]
                    self.following = False
                    self.clamp_camera()
        return True

    def click_sidebar(self, pos):
        for rect, cat in self.filter_buttons:
            if rect.collidepoint(pos):
                if cat == "selected only":
                    self.selected_only = not self.selected_only
                else:
                    self.filters[cat] = not self.filters[cat]
                self.log_scroll = 0
                return
        for rect, agent in self.log_line_rects:
            if rect.collidepoint(pos) and self.sim.agents.alive[agent]:
                self.selected = agent
                self.following = True
                return

    def pan_with_keys(self, dt):
        keys = pygame.key.get_pressed()
        dx = (keys[pygame.K_d] or keys[pygame.K_RIGHT]) - (keys[pygame.K_a] or keys[pygame.K_LEFT])
        dy = (keys[pygame.K_s] or keys[pygame.K_DOWN]) - (keys[pygame.K_w] or keys[pygame.K_UP])
        if dx or dy:
            self.cam_x += dx * PAN_SPEED * dt
            self.cam_y += dy * PAN_SPEED * dt
            self.following = False
            self.clamp_camera()

    # ---------- main loop ----------
    def after_tick(self):
        """Collect marks and log lines from the tick that just happened."""
        tick = self.world.tick
        # Throw away old marks every tick (not only when drawing), so they can't
        # pile up while the screen isn't being drawn (turbo mode)
        self.hit_marks = [m for m in self.hit_marks if 0 <= tick - m[2] < HIT_TICKS]
        self.heart_marks = [m for m in self.heart_marks if 0 <= tick - m[2] < HEART_TICKS]
        self.hit_marks += [(x, y, tick) for x, y in self.sim.agents.hits]
        self.heart_marks += [(x, y, tick) for x, y in self.sim.agents.hearts]
        self.collect_log()

    def frame(self, dt):
        """Everything for one screen update except advancing the world."""
        self.pan_with_keys(dt)
        if self.following and self.selected is not None:
            a = self.sim.agents
            if a.alive[self.selected]:
                self.center_on(a.x[self.selected] + 0.5, a.y[self.selected] + 0.5)
        if self.turbo:
            self.draw_turbo()
        else:
            self.draw_map()
            if self.show_minimap:
                self.draw_minimap()
            self.draw_events()
            self.draw_agent_panel()
            self.draw_eyes()
            if self.show_help:
                self.draw_help()
            self.draw_training()
            self.draw_sidebar()
            self.draw_hud()
        pygame.display.flip()

    def draw(self):
        self.frame(0)

    def draw_turbo(self):
        """Plain screen shown in turbo mode: just the numbers, no world drawing."""
        self.screen.fill((16, 14, 22))
        lines = ["TURBO: training at full speed", "press T to watch again", ""] + self.training_info
        for row, line in enumerate(lines):
            colour = (200, 170, 255) if row == 0 else HUD_TEXT
            label = self.font.render(line, False, colour)
            self.screen.blit(label, (40, 60 + row * 28))

    def run(self):
        """Watch mode: the screen redraws 60 times a second, and the world
        advances at the chosen speed."""
        running = True
        tick_timer = 0.0
        while running:
            running = self.handle_events()
            dt = self.clock.tick(60) / 1000
            if not self.paused:
                tick_timer += dt * SPEEDS[self.speed_index]
                while tick_timer >= 1:
                    self.sim.step()
                    self.after_tick()
                    tick_timer -= 1
            self.frame(dt)
        pygame.quit()
