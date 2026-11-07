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
