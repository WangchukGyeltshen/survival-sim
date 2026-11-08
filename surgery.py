"""Brain surgery: grow an old brain into the current one without losing what it learned.

    python surgery.py runs/run1/latest.pt --name run2
    python train.py --name run2 --resume          then keep training the grown brain

When the world gets new senses or actions (always added at the END of the
lists), the old brain no longer fits. Surgery builds a new, bigger brain and:

  - copies every old weight into the same place in the new brain
  - NEW SENSES: their weights start at zero, so at first the brain ignores
    them (it behaves exactly as before) and learns to use them in training
  - NEW ACTIONS: start with a small chance of being tried (--new-share, 10% in
    total by default), so agents experiment with them without forgetting
    the old ways
  - MEMORY (if the old brain had none): the memory layer starts random, but
    the actor and critic start by ignoring it, so again nothing changes until
    training finds memory useful

Then it checks the result: on the same situations, the new brain must choose
between the old actions exactly as the old brain did.
"""

import argparse
import os

import numpy as np
import torch

from src.simulation import Simulation
from src.observe import observe
from src.brain import Brain, MemoryBank, save_brain


def old_sizes(weights):
    """Work out the sizes of a saved brain from its weights."""
    memory = weights["memory.weight_hh"].shape[1] if "memory.weight_hh" in weights else 0
    return {"hidden": weights["thinking.2.weight"].shape[0],
            "channels": weights["eyes.0.weight"].shape[1],
            "self_size": weights["body.0.weight"].shape[1],
            "actions": weights["actor.weight"].shape[0],
            "memory": memory}


def grow(old_weights, old, new):
    """Copy old weights into the new brain. Returns a description of each change."""
    notes = []
    new_weights = new.state_dict()
    with torch.no_grad():
        for name, target in new_weights.items():
            if name not in old_weights:
                notes.append(f"  new layer      {name:22s} {tuple(target.shape)} (random start)")
                continue
            source = old_weights[name]
            if source.shape == target.shape:
                target.copy_(source)
                continue
            target.zero_()
            target[tuple(slice(0, k) for k in source.shape)] = source
            notes.append(f"  grown          {name:22s} {tuple(source.shape)} -> {tuple(target.shape)}")
    new.load_state_dict(new_weights)
    return notes


def sample_situations(old_brain, sizes, seed, device, ticks=400, every=20):
    """Let the old brain live in a fresh world for a while, and keep copies of
    what agents saw, so the new brain can be checked on realistic situations."""
    sim = Simulation(size=128, agents=200, seed=seed)
    views, feats = [], []
    c, f = sizes["channels"], sizes["self_size"]
    for t in range(ticks):
        idx = np.flatnonzero(sim.agents.alive)
        if len(idx) == 0:
            break
        view, feat = observe(sim, idx)
        if t % every == 0:
            views.append(view)
            feats.append(feat)
        with torch.no_grad():
            v = torch.from_numpy(view[:, :c]).to(device)
            s = torch.from_numpy(feat[:, :f]).to(device)
            acts, _, _, _ = old_brain.act(v, s)
        actions = np.zeros(sim.agents.count, dtype=np.int64)
        actions[idx] = acts.cpu().numpy()
        sim.step(actions)
    return (torch.from_numpy(np.concatenate(views)).to(device),
            torch.from_numpy(np.concatenate(feats)).to(device))


def set_new_actions(new, old_actions, views, feats, share):
    """Give the new actions the scores of an 'average' old action, then shift
    them down until, on average, they get `share` of the chances in total."""
    with torch.no_grad():
        w, b = new.actor.weight, new.actor.bias
        w[old_actions:] = w[:old_actions].mean(dim=0)
        base = b[:old_actions].mean()
        low, high = -30.0, 10.0
        for _ in range(50):                     # binary search for the shift
            mid = (low + high) / 2
            b[old_actions:] = base + mid
            probs = torch.softmax(new(views, feats)[0], dim=1)
            got = probs[:, old_actions:].sum(dim=1).mean().item()
            if got > share:
                high = mid
            else:
                low = mid
        return got


def check(old_brain, new, sizes, views, feats):
    """The new brain, renormalised over the old actions, must match the old brain."""
    with torch.no_grad():
        old_probs = torch.softmax(old_brain(views[:, :sizes["channels"]], feats[:, :sizes["self_size"]])[0], dim=1)
        logits, new_values, _ = new(views, feats)
        new_probs = torch.softmax(logits[:, :sizes["actions"]], dim=1)
        old_values = old_brain(views[:, :sizes["channels"]], feats[:, :sizes["self_size"]])[1]
    return ((old_probs - new_probs).abs().max().item(), (old_values - new_values).abs().max().item())


def main():
    parser = argparse.ArgumentParser(description="Grow an old brain into the current world's brain")
    parser.add_argument("old", help="the old brain, e.g. runs/run1/latest.pt")
    parser.add_argument("--name", default="run2", help="the grown brain goes to runs/<name>/latest.pt")
    parser.add_argument("--new-share", type=float, default=0.10,
                        help="total chance of trying one of the new actions at first (default 0.10)")
    parser.add_argument("--seed", type=int, default=3, help="world used for checking")
    parser.add_argument("--force", action="store_true", help="overwrite runs/<name>/latest.pt")
    args = parser.parse_args()

    out_folder = os.path.join("runs", args.name)
    out_path = os.path.join(out_folder, "latest.pt")
    if os.path.exists(out_path) and not args.force:
        raise SystemExit(f"{out_path} already exists (use --force to overwrite, or pick another --name)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(args.old, map_location=device, weights_only=False)
    weights = data["weights"]
    sizes = data.get("sizes") or old_sizes(weights)
    old_brain = Brain(**sizes).to(device)
    old_brain.load_state_dict(weights)
    new = Brain().to(device)
    for key in ("channels", "self_size", "actions", "memory"):
        if new.sizes[key] < sizes[key]:
            raise SystemExit(f"The new brain has fewer {key} than the old one; surgery can only grow a brain")
    if new.sizes["hidden"] != sizes["hidden"]:
        raise SystemExit("The thinking layers changed size; surgery can't handle that")
    if sizes["memory"] and new.sizes["memory"] != sizes["memory"]:
        raise SystemExit("The memory changed size; surgery can't handle that yet")

    rounds = data.get("extra", {}).get("update", "?")
    print(f"Old brain: {args.old} (round {rounds})")
    for key in ("channels", "self_size", "actions", "memory"):
        print(f"  {key:10s} {sizes[key]:4d} -> {new.sizes[key]:4d}")
    notes = grow(weights, old_brain, new)
    print("\n".join(notes))

    print("Checking on situations from a fresh world...")
    views, feats = sample_situations(old_brain, sizes, args.seed, device)
    if new.sizes["actions"] > sizes["actions"]:
        got = set_new_actions(new, sizes["actions"], views, feats, args.new_share)
        print(f"  new actions get {got:.1%} of the chances in total at first")
    diff_probs, diff_values = check(old_brain, new, sizes, views, feats)
    print(f"  {len(views)} situations: biggest difference from the old brain: "
          f"chances {diff_probs:.2e}, value {diff_values:.2e}")
    if diff_probs > 1e-4 or diff_values > 1e-3:
        raise SystemExit("Surgery check failed: the new brain doesn't behave like the old one")
    print("  OK: the grown brain behaves exactly like the old one")

    os.makedirs(out_folder, exist_ok=True)
    extra = {"update": 0, "total_ticks": 0, "total_samples": 0, "worlds": 1,
             "parent": f"{args.old} round {rounds}"}
    save_brain(new, out_path, extra)
    save_brain(new, os.path.join(out_folder, "brain_round00000.pt"), extra)
    print(f"Saved the grown brain to {out_path}")
    print(f"Next:  python train.py --name {args.name} --resume")


if __name__ == "__main__":
    main()
