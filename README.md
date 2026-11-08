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
| Watch a trained brain | `python play.py --brain runs/run1/latest.pt` |
| Train (with live window) | `python train.py` (continue later with `--resume`) |
| History run (one recorded world) | `python experiment.py --brain runs/run1/latest.pt --name exp1` |
| Control run (untrained brain) | `python experiment.py --blank --name control1` |
| Rewatch a history run | `python replay.py experiments/exp1` |
| Charts for training | `python analyze.py runs/run1` |
| Charts, family tree, heatmap for a history run | `python analyze.py experiments/exp1` |

Charts are saved to `<folder>/charts/`; open `report.html` there in a browser.
