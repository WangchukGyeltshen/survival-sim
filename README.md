# Survival Sim

An experiment to observe how reinforcement learning (RL) agents learn to survive in a primitive, Earth-like world. Agents are not told about resources or dangers; they only know they must survive.

See `docs/DESIGN.md` for the full design.

## Setup (Windows)
```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

The first install line gets the GPU version of PyTorch (the normal one is CPU only).

## Commands

| What | Command |
|---|---|
| Watch random agents | `python play.py` |
| Watch a trained brain | `python play.py --brain runs/run2/latest.pt` |
| Train (with live window) | `python train.py --name run2` (continue later with `--resume`) |
| Train a new run with one shared brain instead of 4 tribe brains | `python train.py --name shared1 --brains 1` |
| Train with a different number of worlds at once (default 4) | `python train.py --name run2 --resume --worlds 6` |
| Grow an old brain to fit the current world (brain surgery) | `python surgery.py runs/run1/latest.pt --name run2` |
| Upgrade a run's brain in place after the world changes | `python surgery.py runs/run2/latest.pt --name run2` |
| History run (one recorded world) | `python experiment.py --brain runs/run2/latest.pt --name exp1` |
| Control run (untrained brain) | `python experiment.py --blank --name control1` |
| Rewatch a history run | `python replay.py experiments/exp1` |
| Charts for training | `python analyze.py runs/run2` |
| Charts, family tree, heatmap for a history run | `python analyze.py experiments/exp1` |

Charts are saved to `<folder>/charts/`; open `report.html` there in a browser.

## Upgrading a brain
Whenever the world gains new senses or actions, older brains no longer fit. `surgery.py` grows an old brain into the new shape without losing what it learned, then checks that it still chooses exactly like the old one. run1 (before Stage 2) was grown into run2 this way. When the world changes again, upgrade the same run in place (its round count is kept and the old brain is backed up), then carry on with `--resume`.

Training runs 4 worlds at once by default; each extra world uses another CPU core, so choose `--worlds` up to about the number of cores your computer has. The window shows world 1.

Each tribe has its own brain. Resuming a run that was saved with one shared brain splits it into 4 tribe brains automatically.
