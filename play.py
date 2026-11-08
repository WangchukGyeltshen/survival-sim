"""Open a window and watch the simulation.

    python play.py                                 agents act randomly
    python play.py --brain runs/run2/latest.pt     agents use a trained brain
"""

import argparse

from src.simulation import Simulation
from src.viewer import Viewer

parser = argparse.ArgumentParser()
parser.add_argument("--brain", help="path to a saved brain (.pt file)")
parser.add_argument("--seed", type=int, default=42, help="which world to generate")
args = parser.parse_args()


def make_sim(seed=None):
    sim = Simulation(size=128, agents=200, seed=seed)
    if policy is not None:
        sim.policy = policy
    return sim


policy = None
if args.brain:
    import torch
    from src.brain import load_brain
    from src.policy import BrainPolicy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    brain, info = load_brain(args.brain, device)
    brain.eval()
    policy = BrainPolicy(brain, device)
    print(f"Loaded brain from {args.brain} (trained for {info.get('update', '?')} rounds)")

viewer = Viewer(make_sim(args.seed))
viewer.new_world_maker = make_sim   # R makes a new world with the same brain
viewer.run()
