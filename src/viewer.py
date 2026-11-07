"""Draws the world in a window using pygame, so it can be watched like a game.

Controls:
    W A S D / arrow keys   move the camera
    mouse drag             move the camera
    mouse wheel            zoom in / out
    click the minimap      jump to that spot
    click an agent         select it and see its stats
    F                      camera follows the selected agent
    V                      show what the selected agent sees
    + / -                  speed up / slow down
    SPACE                  pause / resume
    M                      show / hide minimap
    H                      show / hide help
    R                      new random world
    ESC                    quit
"""

import numpy as np
import pygame

from src.world import POND, TICKS_PER_DAY, DAYLIGHT_TICKS
from src.sprites import (SpriteSheet, PIXELS, minimap_colour, agent_sprites,
                         animal_sprites, WOLF_EYE, disaster_sprites, farm_sprites)
from src.world import FARMLAND, CROP_GROW_TICKS
from src.agents import ITEMS, WEAPON_NAMES, HANDS, GENE_NAMES, ADULT_AGE
from src.observe import observe, CH, VIEW_SIZE, TERRAIN_ORDER
from src.animals import RABBIT, DEER, WOLF, BEAR
from src.simulation import Simulation
from src.agents import ACTION_NAMES, DEATH_CAUSES, TICKS_PER_YEAR, MALE

WINDOW_W, WINDOW_H = 1024, 640   # size of the map view in screen pixels
HUD_HEIGHT = 36                  # info bar at the top
ZOOMS = [PIXELS * n for n in (1, 2, 3, 4)]   # tile sizes: 8, 16, 24, 32 pixels
START_ZOOM = 1                   # 16 pixel tiles
PAN_SPEED = 600                  # camera speed in screen pixels per second
MINIMAP_SCALE = 1                # minimap pixels per tile

SPEEDS = [1, 2, 5, 10, 20, 50, 100, 300]   # ticks per second
GRAVE_TICKS = TICKS_PER_DAY      # bones stay visible for one day
SELECT_COLOUR = (255, 230, 90)
BAR_COLOURS = {
    "health": (214, 60, 70),
    "food": (230, 150, 60),
    "water": (70, 150, 230),
    "energy": (240, 210, 80),
    "warmth": (240, 120, 160),
}

HUD_BG = (24, 20, 28)
HUD_TEXT = (236, 228, 210)
SUN = (250, 204, 70)
MOON = (200, 210, 235)
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
    "F               follow selected",
    "V               agent's eyes",
    "+ / -           speed",
    "SPACE           pause",
    "M               minimap",
    "R               new world",
    "T               turbo (training)",
    "H               close help",
    "ESC             quit",
]


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


class Viewer:
    def __init__(self, sim):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H + HUD_HEIGHT))
        pygame.display.set_caption("Survival Sim")
        self.font = pygame.font.SysFont("consolas", 18, bold=True)
        self.small_font = pygame.font.SysFont("consolas", 15, bold=True)
        self.clock = pygame.time.Clock()
        self.sheets = {size: SpriteSheet(size) for size in ZOOMS}
        self.people = {size: agent_sprites(size) for size in ZOOMS}
        self.beasts = {size: animal_sprites(size) for size in ZOOMS}
        self.hazards = {size: disaster_sprites(size) for size in ZOOMS}
        self.farm = {size: farm_sprites(size) for size in ZOOMS}
        self.hit_marks = []      # [x, y, frames left] red slashes where hits landed
        self.heart_marks = []    # [x, y, frames left] hearts where a child was conceived
        self.show_eyes = False
        # Training mode (set by train.py)
        self.training = False
        self.turbo = False         # T: stop drawing so training runs at full speed
        self.training_info = []    # lines of text shown in the training panel
        self.new_world_maker = lambda: Simulation(size=self.world.size)
        self.flash = 0           # frames of lightning flash left
        self.rain = np.random.default_rng(1).random((160, 2))   # raindrop positions
        self.zoom = START_ZOOM
        self.speed_index = 3   # start at 10 ticks per second
        self.paused = False
        self.show_minimap = True
        self.show_help = False
        self.mouse_down_at = None
        self.dragging = False
        self.selected = None     # index of the selected agent
        self.following = False
        self.night = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
        self.set_sim(sim)

    @property
    def tile(self):
        return ZOOMS[self.zoom]

    def set_sim(self, sim):
        self.sim = sim
        self.world = world = sim.world
        self.selected = None
        self.following = False
        # Pick a random look for each grass tile once, so the map doesn't flicker
        self.variants = np.random.default_rng(0).integers(0, 10, (world.size, world.size))
        self.minimap = None
        self.minimap_season = None
        # Start with the camera in the middle of the world
        self.cam_x = world.size * self.tile / 2 - WINDOW_W / 2
        self.cam_y = world.size * self.tile / 2 - WINDOW_H / 2
        self.clamp_camera()

    # ---------- camera ----------
    def clamp_camera(self):
        """Stop the camera from moving past the edges of the world."""
        world_px = self.world.size * self.tile
        self.cam_x = min(max(self.cam_x, 0), max(world_px - WINDOW_W, 0))
        self.cam_y = min(max(self.cam_y, 0), max(world_px - WINDOW_H, 0))

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
        self.cam_x = tile_x * self.tile - WINDOW_W / 2
        self.cam_y = tile_y * self.tile - WINDOW_H / 2
        self.clamp_camera()

    # ---------- drawing ----------
    def draw_map(self):
        w = self.world
        tile = self.tile
        sheet = self.sheets[tile]
        winter = w.season == "winter"
        frame = (pygame.time.get_ticks() // 500) % 2   # water ripples every half second

        # Only draw the tiles the camera can see
        first_x, first_y = int(self.cam_x // tile), int(self.cam_y // tile)
        last_x = min(first_x + WINDOW_W // tile + 2, w.size)
        last_y = min(first_y + WINDOW_H // tile + 2, w.size)

        self.screen.set_clip((0, HUD_HEIGHT, WINDOW_W, WINDOW_H))
        ash, fire_frames, flood = self.hazards[tile]
        crops, pile_img, _ = self.farm[tile]
        jobs, overlays, fires = [], [], []
        berry_tiles = []
        for y in range(first_y, last_y):
            py = HUD_HEIGHT + y * tile - int(self.cam_y)
            for x in range(first_x, last_x):
                kind = w.terrain[y, x]
                water_frame = frame if kind == POND else frame * 2
                px = x * tile - int(self.cam_x)
                if w.burnt[y, x] > 0:
                    jobs.append((ash, (px, py)))
                else:
                    jobs.append((sheet.get(kind, self.variants[y, x], winter, water_frame), (px, py)))
                if kind == FARMLAND:
                    if w.berries[y, x]:
                        overlays.append((crops[2], (px, py)))          # ripe
                    else:
                        stage = 0 if w.crop[y, x] > CROP_GROW_TICKS // 2 else 1
                        overlays.append((crops[stage], (px, py)))
                elif w.berries[y, x]:
                    berry_tiles.append((px, py, int(w.berries[y, x]), bool(w.poison[y, x])))
                if (x, y) in w.piles:
                    overlays.append((pile_img, (px, py)))
                if w.flood[y, x] > 0:
                    overlays.append((flood, (px, py)))
                if w.fire[y, x] > 0:
                    fires.append((fire_frames[(frame + x + y) % 2], (px, py)))
        self.screen.blits(jobs, doreturn=False)
        for px, py, count, poison in berry_tiles:
            sheet.draw_berries(self.screen, px, py, count, poison)
        self.screen.blits(overlays, doreturn=False)
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
                x = int((fx * WINDOW_W + t * (20 if snow else -120)) % WINDOW_W)
                y = HUD_HEIGHT + int((fy * WINDOW_H + t * speed) % WINDOW_H)
                if snow:
                    pygame.draw.rect(self.screen, (240, 244, 250), (x, y, 3, 3))
                else:
                    pygame.draw.line(self.screen, (170, 190, 230), (x, y), (x - 3, y + 10), 1)
        if self.flash > 0:
            self.night.fill((255, 255, 255, 30 * self.flash))
            self.screen.blit(self.night, (0, HUD_HEIGHT))
            self.flash -= 1

    def draw_events(self):
        """Recent disaster messages (yellow) and first discoveries (green),
        top-left under the info bar."""
        tick = self.world.tick
        events = [(t, m, (255, 214, 120)) for t, m in self.sim.disasters.events[-4:]]
        events += [(t, f"Day {t // TICKS_PER_DAY}: {m}", (140, 230, 120))
                   for t, m in self.sim.discoveries[-4:]]
        events.sort(key=lambda e: e[0])
        recent = [e for e in events if tick - e[0] < 300][-5:]
        for row, (t, message, colour) in enumerate(recent):
            label = self.small_font.render(message, False, colour)
            box = label.get_rect(topleft=(10, HUD_HEIGHT + 10 + row * 24)).inflate(12, 6)
            panel = pygame.Surface(box.size, pygame.SRCALPHA)
            panel.fill((24, 20, 28, 200))
            self.screen.blit(panel, box)
            self.screen.blit(label, (10, HUD_HEIGHT + 10 + row * 24))

    def draw_wolf_eyes(self, first_x, first_y, last_x, last_y):
        """Wolves' eyes glow in the dark, drawn on top of the night layer."""
        an = self.sim.animals
        size = self.tile // PIXELS
        row, col = WOLF_EYE
        wolves = an.alive & (an.species == WOLF) & (an.x >= first_x) & (an.x < last_x) \
            & (an.y >= first_y) & (an.y < last_y)
        for i in np.flatnonzero(wolves):
            px, py = self.to_screen(an.x[i], an.y[i])
            pygame.draw.rect(self.screen, (255, 220, 60), (px + col * size, py + row * size, size, size))

    def draw_agents(self, first_x, first_y, last_x, last_y):
        a = self.sim.agents
        tile = self.tile
        people, bones, heart = self.people[tile]
        in_view = (a.x >= first_x) & (a.x < last_x) & (a.y >= first_y) & (a.y < last_y)

        # Bones where agents died recently, fading away
        recent = ~a.alive & in_view & (a.death_tick >= 0) & \
            (self.world.tick - a.death_tick < GRAVE_TICKS)
        for i in np.flatnonzero(recent):
            age = self.world.tick - a.death_tick[i]
            bones.set_alpha(int(255 * (1 - age / GRAVE_TICKS)))
            self.screen.blit(bones, self.to_screen(a.x[i], a.y[i]))
        bones.set_alpha(255)

        an = self.sim.animals
        beasts = self.beasts[tile]
        animals_in_view = an.alive & (an.x >= first_x) & (an.x < last_x) & \
            (an.y >= first_y) & (an.y < last_y)
        for i in np.flatnonzero(animals_in_view):
            self.screen.blit(beasts[an.species[i]], self.to_screen(an.x[i], an.y[i]))

        weapons = self.farm[tile][2]
        adult_age = ADULT_AGE * TICKS_PER_YEAR
        for i in np.flatnonzero(a.alive & in_view):
            pos = self.to_screen(a.x[i], a.y[i])
            kind = "child" if a.age[i] < adult_age else ("pregnant" if a.pregnant[i] else "adult")
            self.screen.blit(people[(a.sex[i], kind)], pos)
            if a.weapon[i] != HANDS:
                self.screen.blit(weapons[a.weapon[i]], pos)

        # Red slashes where hits landed
        for mark in self.hit_marks:
            px, py = self.to_screen(mark[0], mark[1])
            pygame.draw.line(self.screen, (240, 40, 40), (px + 2, py + 2), (px + tile - 3, py + tile - 3), 3)
            mark[2] -= 1
        self.hit_marks = [m for m in self.hit_marks if m[2] > 0]
        for mark in self.heart_marks:
            px, py = self.to_screen(mark[0], mark[1])
            self.screen.blit(heart, (px, py - tile // 2 - (40 - mark[2]) // 4))   # floats up
            mark[2] -= 1
        self.heart_marks = [m for m in self.heart_marks if m[2] > 0]

        if self.selected is not None and a.alive[self.selected]:
            px, py = self.to_screen(a.x[self.selected], a.y[self.selected])
            pygame.draw.rect(self.screen, SELECT_COLOUR, (px - 1, py - 1, tile + 2, tile + 2), 2)

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

    def draw_agent_panel(self):
        """Stats of the selected agent, bottom-left."""
        i = self.selected
        if i is None:
            return
        a = self.sim.agents
        height = 268 if a.alive[i] else 56
        box = pygame.Rect(10, HUD_HEIGHT + WINDOW_H - height - 10, 410, height)
        panel = pygame.Surface(box.size, pygame.SRCALPHA)
        panel.fill((24, 20, 28, 225))
        self.screen.blit(panel, box)
        pygame.draw.rect(self.screen, (60, 52, 70), box, 2)

        sex = "Male" if a.sex[i] == MALE else "Female"
        years = a.age[i] / TICKS_PER_YEAR
        title = f"#{i}  {sex}  age {years:.1f}y  gen {a.generation[i]}"
        label = self.small_font.render(title, False, HUD_TEXT)
        self.screen.blit(label, (box.x + 10, box.y + 8))
        tags = []
        if a.alive[i] and a.sick[i] > 0:
            tags.append(("SICK", (150, 220, 90)))
        if a.alive[i] and a.pregnant[i] > 0:
            tags.append((f"PREGNANT {a.pregnant[i] / TICKS_PER_DAY:.1f}d", (240, 120, 160)))
        tx = box.x + 18 + label.get_width()
        for text, colour in tags:
            tag = self.small_font.render(text, False, colour)
            self.screen.blit(tag, (tx, box.y + 8))
            tx += tag.get_width() + 8

        if not a.alive[i]:
            cause = DEATH_CAUSES[a.cause_of_death[i]]
            label = self.small_font.render(f"DEAD ({cause})", False, (214, 60, 70))
            self.screen.blit(label, (box.x + 10, box.y + 30))
            return

        y = box.y + 32
        for name, colour in BAR_COLOURS.items():
            value = getattr(a, name)[i]
            self.screen.blit(self.small_font.render(name, False, HUD_TEXT), (box.x + 10, y))
            pygame.draw.rect(self.screen, (50, 44, 56), (box.x + 84, y + 3, 150, 11))
            pygame.draw.rect(self.screen, colour, (box.x + 84, y + 3, int(1.5 * value), 11))
            y += 20
        doing = ACTION_NAMES[a.last_action[i]]
        follow = "  [following]" if self.following else ""
        label = self.small_font.render(f"doing: {doing}{follow}", False, HUD_TEXT)
        self.screen.blit(label, (box.x + 10, y + 4))

        weapon = WEAPON_NAMES[a.weapon[i]]
        if a.weapon[i] != HANDS:
            weapon += f" ({a.weapon_uses[i]} uses)"
        label = self.small_font.render(f"weapon: {weapon}", False, HUD_TEXT)
        self.screen.blit(label, (box.x + 10, y + 24))
        carried = ", ".join(f"{a.inv[item][i]} {item}" for item in ITEMS if a.inv[item][i])
        label = self.small_font.render(f"carry: {carried or 'nothing'}", False, HUD_TEXT)
        self.screen.blit(label, (box.x + 10, y + 44))
        genes = "  ".join(f"{g[:3]} {a.genes[g][i]:.2f}" for g in GENE_NAMES)
        label = self.small_font.render(f"genes: {genes}", False, (180, 200, 240))
        self.screen.blit(label, (box.x + 10, y + 64))
        family = f"children: {a.children[i]}   kills: {a.kills[i]}"
        self.screen.blit(self.small_font.render(family, False, (170, 160, 180)), (box.x + 10, y + 84))

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
        return pygame.Rect(WINDOW_W - size - 10, HUD_HEIGHT + WINDOW_H - size - 10, size, size)

    def draw_minimap(self):
        if self.minimap is None or self.minimap_season != self.world.season:
            self.build_minimap()
        rect = self.minimap_rect()
        pygame.draw.rect(self.screen, HUD_BG, rect.inflate(6, 6))
        self.screen.blit(self.minimap, rect)
        a = self.sim.agents
        for i in np.flatnonzero(a.alive):
            self.screen.set_at((rect.x + int(a.x[i] * MINIMAP_SCALE),
                                rect.y + int(a.y[i] * MINIMAP_SCALE)), (255, 255, 255))
        an = self.sim.animals
        for i in np.flatnonzero(an.alive & np.isin(an.species, (WOLF, BEAR))):
            self.screen.set_at((rect.x + int(an.x[i] * MINIMAP_SCALE),
                                rect.y + int(an.y[i] * MINIMAP_SCALE)), (230, 40, 40))
        # White box showing what the camera sees
        scale = MINIMAP_SCALE / self.tile
        view = pygame.Rect(rect.x + self.cam_x * scale, rect.y + self.cam_y * scale,
                           WINDOW_W * scale, WINDOW_H * scale).clip(rect)
        pygame.draw.rect(self.screen, (255, 255, 255), view, 1)

    def draw_hud(self):
        w = self.world
        pygame.draw.rect(self.screen, HUD_BG, (0, 0, WINDOW_W, HUD_HEIGHT))
        pygame.draw.rect(self.screen, (60, 52, 70), (0, HUD_HEIGHT - 3, WINDOW_W, 3))

        # Sun or moon icon (pixel squares)
        if w.is_night:
            pygame.draw.rect(self.screen, MOON, (10, 9, 15, 15))
            pygame.draw.rect(self.screen, HUD_BG, (17, 6, 12, 12))
        else:
            pygame.draw.rect(self.screen, SUN, (10, 10, 15, 15))
            for dx, dy in ((-6, 5), (19, 5), (5, -6), (5, 19)):
                pygame.draw.rect(self.screen, SUN, (10 + dx, 10 + dy, 3, 3))

        x = 40
        x = self.text(f"DAY {w.day}", x, HUD_TEXT)
        x = self.text(w.season.upper(), x + 20, SEASON_COLOURS[w.season])

        # Berry counter with a little berry icon
        pygame.draw.rect(self.screen, (214, 40, 64), (x + 20, 13, 9, 9))
        pygame.draw.rect(self.screen, (78, 150, 74), (x + 23, 9, 3, 4))
        x = self.text(str(int(w.berries.sum())), x + 34, HUD_TEXT)

        a = self.sim.agents
        males = int((a.alive & (a.sex == MALE)).sum())
        x = self.text(f"ALIVE {a.population}", x + 20, HUD_TEXT)
        x = self.text(f"({males}M {a.population - males}F)", x + 6, (170, 160, 180))
        dead = sum(self.sim.deaths.values())
        x = self.text(f"DEAD {dead}", x + 16, (214, 90, 90))
        x = self.text(f"BORN {a.births}", x + 16, (240, 120, 160))
        top_gen = int(a.generation[a.alive].max()) if a.population else 0
        x = self.text(f"GEN {top_gen}", x + 16, (180, 200, 240))

        # Active disasters
        if w.storm > 0:
            x = self.text("BLIZZARD" if w.season == "winter" else "STORM", x + 20, (170, 190, 230))
        if (w.flood > 0).any():
            x = self.text("FLOOD", x + 14, (90, 150, 230))
        if (w.fire > 0).any():
            x = self.text("FIRE", x + 14, (255, 140, 40))

        speed = "MAX" if self.training and self.speed_index == len(SPEEDS) - 1 else f"x{SPEEDS[self.speed_index]}"
        status = "PAUSED" if self.paused else speed
        right = f"{status}   H: help"
        label = self.font.render(right, False, HUD_TEXT)
        self.screen.blit(label, (WINDOW_W - label.get_width() - 12, 8))

    def draw_census(self):
        """Animal counts, top-right under the info bar."""
        an = self.sim.animals
        beasts = self.beasts[ZOOMS[1]]
        box = pygame.Rect(WINDOW_W - 138, HUD_HEIGHT + 8, 128, 4 * 22 + 10)
        panel = pygame.Surface(box.size, pygame.SRCALPHA)
        panel.fill((24, 20, 28, 200))
        self.screen.blit(panel, box)
        for row, kind in enumerate((RABBIT, DEER, WOLF, BEAR)):
            y = box.y + 5 + row * 22
            self.screen.blit(beasts[kind], (box.x + 6, y + 2))
            name = ["Rabbits", "Deer", "Wolves", "Bears"][row]
            label = self.small_font.render(f"{name} {an.count(kind)}", False, HUD_TEXT)
            self.screen.blit(label, (box.x + 28, y + 2))

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
        cell = 10
        size = VIEW_SIZE * cell
        box = pygame.Rect(WINDOW_W - size - 20, HUD_HEIGHT + 120, size + 12, size + 34)
        pygame.draw.rect(self.screen, HUD_BG, box)
        pygame.draw.rect(self.screen, (60, 52, 70), box, 2)
        label = self.small_font.render(f"What #{i} sees", False, HUD_TEXT)
        self.screen.blit(label, (box.x + 6, box.y + 4))
        ox, oy = box.x + 6, box.y + 26
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
                for name, mark in self.EYE_MARKS:
                    if v[CH[name], ty, tx] > 0:
                        pygame.draw.rect(self.screen, mark, (rect[0] + 2, rect[1] + 2, cell - 4, cell - 4))
                        break
        c = VIEW_SIZE // 2
        pygame.draw.rect(self.screen, (255, 230, 90), (ox + c * cell, oy + c * cell, cell, cell), 1)

    def draw_help(self):
        line_h = 20
        box = pygame.Rect(16, HUD_HEIGHT + 16, 300, len(HELP_LINES) * line_h + 20)
        panel = pygame.Surface(box.size, pygame.SRCALPHA)
        panel.fill((24, 20, 28, 220))
        self.screen.blit(panel, box)
        pygame.draw.rect(self.screen, (60, 52, 70), box, 2)
        for i, line in enumerate(HELP_LINES):
            label = self.small_font.render(line, False, HUD_TEXT)
            self.screen.blit(label, (box.x + 12, box.y + 10 + i * line_h))

    def text(self, string, x, colour):
        """Draw text without smoothing (keeps the pixel look). Returns the end x."""
        label = self.font.render(string, False, colour)
        self.screen.blit(label, (x, 8))
        return x + label.get_width()

    def draw(self):
        self.frame(0)

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
                if event.key == pygame.K_f and self.selected is not None:
                    self.following = not self.following

            if event.type == pygame.MOUSEWHEEL:
                mx, my = pygame.mouse.get_pos()
                self.set_zoom(self.zoom + event.y, (mx, my - HUD_HEIGHT))

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
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

    def pan_with_keys(self, dt):
        keys = pygame.key.get_pressed()
        dx = (keys[pygame.K_d] or keys[pygame.K_RIGHT]) - (keys[pygame.K_a] or keys[pygame.K_LEFT])
        dy = (keys[pygame.K_s] or keys[pygame.K_DOWN]) - (keys[pygame.K_w] or keys[pygame.K_UP])
        if dx or dy:
            self.cam_x += dx * PAN_SPEED * dt
            self.cam_y += dy * PAN_SPEED * dt
            self.clamp_camera()

    def after_tick(self):
        """Collect hit and heart marks from the tick that just happened."""
        self.hit_marks += [[x, y, 12] for x, y in self.sim.agents.hits]
        self.heart_marks += [[x, y, 40] for x, y in self.sim.agents.hearts]

    def frame(self, dt):
        """Everything for one screen update except advancing the world."""
        self.pan_with_keys(dt)
        if self.following and self.selected is not None:
            a = self.sim.agents
            self.center_on(a.x[self.selected] + 0.5, a.y[self.selected] + 0.5)
        if self.turbo:
            self.draw_turbo()
        else:
            self.draw_map()
            if self.show_minimap:
                self.draw_minimap()
            self.draw_census()
            self.draw_events()
            self.draw_agent_panel()
            self.draw_eyes()
            if self.show_help:
                self.draw_help()
            self.draw_training()
            self.draw_hud()
        pygame.display.flip()

    def draw_training(self):
        """Training progress panel (only while training)."""
        if not self.training_info:
            return
        line_h = 20
        width = 330
        box = pygame.Rect(WINDOW_W - width - 180, HUD_HEIGHT + 8, width, len(self.training_info) * line_h + 12)
        panel = pygame.Surface(box.size, pygame.SRCALPHA)
        panel.fill((24, 20, 28, 215))
        self.screen.blit(panel, box)
        pygame.draw.rect(self.screen, (90, 70, 140), box, 2)
        for row, line in enumerate(self.training_info):
            colour = (200, 170, 255) if row == 0 else HUD_TEXT
            label = self.small_font.render(line, False, colour)
            self.screen.blit(label, (box.x + 10, box.y + 6 + row * line_h))

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
