"""Open a window and watch the simulation run like a game."""

from src.simulation import Simulation
from src.viewer import Viewer

sim = Simulation(size=128, agents=200, seed=42)
Viewer(sim).run()
