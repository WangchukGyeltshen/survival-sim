"""Compare runs side by side: the same charts, one line per run.

    python compare.py runs/run2 runs/shared1                     training runs
    python compare.py experiments/exp1 experiments/control1      history runs
    python compare.py runs/run2 runs/shared1 --name tribes_vs_shared

Give 2 or more folders of the same kind. Charts and a report.html go to
comparisons/<name>/ (open report.html in your browser).

TRAINING RUNS are lined up by how much experience they have had (decisions
trained, in millions), not by round number, so runs with different settings
are compared fairly. HISTORY RUNS are lined up by day.

A fair comparison changes ONE thing. Some ideas:
    tribe brains vs one shared brain:
        python train.py --name tribes1                 (4 brains, the default)
        python train.py --name shared1 --brains 1
    behaviour genes on vs off:
        python train.py --name genes1
        python train.py --name nogenes1 --no-temperament
    trained vs untrained, in the very same world (same --seed):
        python experiment.py --brain runs/run2/latest.pt --name exp1 --seed 7
        python experiment.py --blank --name control1 --seed 7
Luck plays a big part in any single run, so differences only count if they
hold up over a few runs (for history runs: a few different seeds).
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import read_csv, smooth, save, column, write_report, CAUSE_COLOURS   # noqa: E402
from src.agents import DEATH_CAUSES, BEHAVIOUR_GENES, GENE_NAMES                  # noqa: E402

# Colour-blind-friendly line colours, one per run
RUN_COLOURS = ["#2f6fdb", "#e0652b", "#2a9d6a", "#b8459c", "#c9a227", "#5b5f6b"]
CAUSES = [c for c in DEATH_CAUSES[1:]]


def kind_of(folder):
    if os.path.exists(os.path.join(folder, "log.csv")):
        return "training"
    if os.path.exists(os.path.join(folder, "daily.csv")):
        return "history"
    raise SystemExit(f"{folder} has neither log.csv (training) nor daily.csv (history run)")


def label_of(folder):
    return os.path.basename(os.path.normpath(folder))


def settings_of(folder, kind):
    """A short description of a run's settings, for the report."""
    if kind == "history":
        path = os.path.join(folder, "info.json")
        if not os.path.exists(path):
            return ""
        info = json.load(open(path))
        parts = [f"seed {info.get('seed')}", f"brain {info.get('brain') or 'blank'}",
                 "learning" if info.get("learning") else "frozen"]
        if "brains" in info:
            parts.append(f"{info['brains']} brain(s)")
        if info.get("temperament") is False:
            parts.append("temperament off")
        return ", ".join(parts)
    path = os.path.join(folder, "latest.pt")
    if not os.path.exists(path):
        return ""
    import torch
    data = torch.load(path, map_location="cpu", weights_only=False)
    extra = data.get("extra", {})
    brains = len(data["brains"]) if "brains" in data else 1
    parts = [f"round {extra.get('update', '?')}", f"{brains} brain(s)"]
    if extra.get("temperament") is False:
        parts.append("temperament off")
    if extra.get("parent"):
        parts.append(f"grown from {extra['parent']}")
    return ", ".join(parts)


def line_chart(runs, x_key, x_label, y_func, title, y_label, out, charts, name, window):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for k, (label, d) in enumerate(runs):
        x = x_key(d)
        ax.plot(x, smooth(y_func(d), window), lw=2, color=RUN_COLOURS[k % len(RUN_COLOURS)], label=label)
    ax.set(title=title, xlabel=x_label, ylabel=y_label)
    ax.legend()
    save(fig, out, name, charts)


def cause_bars(runs, prefix, title, out, charts, name, last_share):
    """Share of each cause of death over the last part of each run."""
    shares = []
    for label, d in runs:
        n = len(next(iter(d.values())))
        start = int(n * (1 - last_share))
        counts = np.array([np.nansum(column(d, f"{prefix}{c}", n)[start:]) for c in CAUSES], dtype=float)
        shares.append(counts / max(counts.sum(), 1))
    shares = np.array(shares)
    fig, ax = plt.subplots(figsize=(10, 1.2 + 0.7 * len(runs)))
    left = np.zeros(len(runs))
    labels = [label for label, _ in runs]
    for j, cause in enumerate(CAUSES):
        if shares[:, j].max() < 0.005:
            continue
        ax.barh(labels, shares[:, j], left=left, color=CAUSE_COLOURS.get(cause, "#cccccc"), label=cause)
        for i, value in enumerate(shares[:, j]):
            if value > 0.06:
                ax.text(left[i] + value / 2, i, f"{value:.0%}", ha="center", va="center", fontsize=9, color="white")
        left += shares[:, j]
    ax.set(title=title, xlim=(0, 1))
    ax.invert_yaxis()
    ax.legend(ncol=6, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    ax.grid(False)
    save(fig, out, name, charts)


def decisions_per_round(d):
    """How many decisions each round of a training run collected."""
    per = np.diff(d["samples"], prepend=np.nan)
    if len(per) > 1:
        per[0] = per[1]          # the first round has nothing before it; assume it was typical
    return np.maximum(np.nan_to_num(per, nan=1.0), 1)


# ======================================================================
def compare_training(folders, out):
    runs = [(label_of(f), read_csv(os.path.join(f, "log.csv"))) for f in folders]
    charts = []
    longest = max(len(d["round"]) for _, d in runs)
    window = max(1, longest // 60)
    x = lambda d: d["samples"] / 1e6
    xl = "decisions trained (millions)"
    line_chart(runs, x, xl, lambda d: d["avg_days_survived"], "Average days survived (recent deaths)",
               "days", out, charts, "survival.png", window)
    line_chart(runs, x, xl, lambda d: d["reward_per_day"], "Reward per agent per day", "reward",
               out, charts, "reward.png", window)
    line_chart(runs, x, xl, lambda d: d["curiosity"] * 100, "Curiosity (how unsure the brains are)",
               "% of the most unsure possible", out, charts, "curiosity.png", window)

    line_chart(runs, x, xl, lambda d: d["births"] / decisions_per_round(d) * 10_000,
               "Births per 10,000 decisions", "births", out, charts, "births.png", window)
    cause_bars(runs, "died_", "Causes of death (last 20% of each run)", out, charts, "causes_of_death.png", 0.2)
    if all("seconds" in d for _, d in runs):
        line_chart(runs, x, xl, lambda d: decisions_per_round(d) / np.maximum(d["seconds"], 0.1),
                   "Training speed", "decisions per second", out, charts, "speed.png", window)

    summary = []
    for f, (label, d) in zip(folders, runs):
        tail = slice(int(len(d["round"]) * 0.9), None)       # the last 10% of rounds
        summary.append(f"<b>{label}</b> ({settings_of(f, 'training')}): {d['samples'][-1] / 1e6:.1f} million decisions; "
                       f"last 10%: survived {np.nanmean(d['avg_days_survived'][tail]):.2f} days on average, "
                       f"reward/day {np.nanmean(d['reward_per_day'][tail]):.2f}, "
                       f"curiosity {np.nanmean(d['curiosity'][tail]):.0%}")
    return charts, summary


def compare_history(folders, out):
    runs = [(label_of(f), read_csv(os.path.join(f, "daily.csv"))) for f in folders]
    charts = []
    longest = max(len(d["day"]) for _, d in runs)
    window = max(1, longest // 50)
    x = lambda d: d["day"]
    n = lambda d: len(d["day"])
    line_chart(runs, x, "day", lambda d: d["population"], "Population", "alive", out, charts,
               "population.png", 1)
    line_chart(runs, x, "day", lambda d: d["births"], "Births per day", "births", out, charts,
               "births.png", window)
    line_chart(runs, x, "day", lambda d: d["deaths"], "Deaths per day", "deaths", out, charts,
               "deaths.png", window)
    line_chart(runs, x, "day", lambda d: d["max_generation"], "Highest generation reached", "generation",
               out, charts, "generations.png", 1)
    line_chart(runs, x, "day", lambda d: d["armed"] / np.maximum(d["population"], 1) * 100,
               "Share of people carrying a weapon", "%", out, charts, "armed.png", window)
    line_chart(runs, x, "day", lambda d: column(d, "killed", n(d)), "People killed by people, per day",
               "killings", out, charts, "violence.png", window)
    line_chart(runs, x, "day", lambda d: sum(column(d, k, n(d)) for k in ("hunted", "harvested", "planted")),
               "Food work per day (hunting, planting, harvesting)", "times", out, charts, "food_work.png", window)
    line_chart(runs, x, "day", lambda d: sum(v for k, v in d.items() if k.startswith("built_")),
               "Structures standing", "structures", out, charts, "structures.png", 1)
    line_chart(runs, x, "day", lambda d: column(d, "gave", n(d)) + column(d, "stored", n(d)),
               "Sharing per day (giving and storing)", "times", out, charts, "sharing.png", window)
    cause_bars(runs, "died_", "Causes of death (whole run)", out, charts, "causes_of_death.png", 1.0)

    genes = [g for g in BEHAVIOUR_GENES if all(f"gene_{g}" in d for _, d in runs)]
    for title, names, file in (("Body genes", GENE_NAMES, "genes_body.png"),
                               ("Behaviour genes (temperament)", genes, "genes_behaviour.png")):
        names = [g for g in names if all(f"gene_{g}" in d for _, d in runs)]
        if not names:
            continue
        fig, axes = plt.subplots(1, len(names), figsize=(3.4 * len(names), 3.6), squeeze=False)
        for ax, g in zip(axes[0], names):
            for k, (label, d) in enumerate(runs):
                ax.plot(d["day"], d[f"gene_{g}"], lw=2, color=RUN_COLOURS[k % len(RUN_COLOURS)], label=label)
            ax.set(title=g, xlabel="day")
        axes[0][0].legend(fontsize=8)
        fig.suptitle(f"{title}: average over everyone alive")
        save(fig, out, file, charts)

    summary = []
    for f, (label, d) in zip(folders, runs):
        summary.append(f"<b>{label}</b> ({settings_of(f, 'history')}): lasted {int(d['day'][-1])} days, "
                       f"peak population {int(np.nanmax(d['population']))}, "
                       f"final population {int(d['population'][-1])}, "
                       f"{int(np.nansum(d['births']))} births, "
                       f"reached generation {int(np.nanmax(d['max_generation']))}")
    return charts, summary


def main():
    parser = argparse.ArgumentParser(description="Compare runs side by side")
    parser.add_argument("folders", nargs="+", help="2 or more runs/<name> or experiments/<name> folders")
    parser.add_argument("--name", help="results go to comparisons/<name>/ (default: the run names joined)")
    args = parser.parse_args()
    if len(args.folders) < 2:
        raise SystemExit("Give at least 2 folders to compare")
    kinds = {kind_of(f) for f in args.folders}
    if len(kinds) > 1:
        raise SystemExit("Compare training runs with training runs, and history runs with history runs")
    kind = kinds.pop()
    name = args.name or "_vs_".join(label_of(f) for f in args.folders)
    out = os.path.join("comparisons", name)
    os.makedirs(out, exist_ok=True)
    charts, summary = (compare_training if kind == "training" else compare_history)(args.folders, out)
    title = f"Comparison of {kind} runs: " + " vs ".join(label_of(f) for f in args.folders)
    path = write_report(out, out, charts, summary, title)
    print(f"{len(charts)} charts saved in {out}")
    print(f"Open this in your browser: {path}")


if __name__ == "__main__":
    main()
