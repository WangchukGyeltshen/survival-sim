"""Generate a world and print it as text (quick check without graphics)."""

from src.world import World

world = World(size=128, seed=42)
print(world.render_text())
