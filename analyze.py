"""Turn the numbers from training or a history run into charts.

    python analyze.py runs/run1           charts of how training went
    python analyze.py experiments/exp1    charts, family tree and heatmap of a history run

Charts are saved as pictures in <folder>/charts/, plus a report.html that
shows them all on one page (open it in your browser).
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")            # draw to files, no window
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.agents import DEATH_CAUSES, ACTION_NAMES, GENE_NAMES   # noqa: E402
from src.sprites import minimap_colour                           # noqa: E402

plt.rcParams.update({"figure.facecolor": "white", "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 11})
CAUSE_COLOURS = {
    "starvation": "#d9822b", "thirst": "#3b82c4", "wolf": "#6b6b78", "bear": "#7a4b2a",
    "cold": "#8ec5e8", "poison": "#9b59d0", "sickness": "#7fbf3f", "drowned": "#1f4e8c",
    "fell": "#a08070", "flood": "#4fa3d9", "fire": "#e8501c", "lightning": "#f2c12e",
    "old age": "#999999", "deer": "#c79a5b", "killed by person": "#c0392b", "childbirth": "#e88fb0",
}


def read_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) if r[k] not in ("", None) else np.nan for r in rows])
            if k not in ("season",) else [r[k] for r in rows] for k in (rows[0] if rows else [])}


def smooth(values, window):
    """Rolling average, so noisy lines become readable trends."""
    values = np.asarray(values, dtype=float)
    if len(values) < window or window <= 1:
        return values
    kernel = np.ones(window) / window
    padded = np.concatenate([np.full(window - 1, values[0]), values])
    return np.convolve(padded, kernel, mode="valid")


def save(fig, folder, name, charts):
    path = os.path.join(folder, name)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    charts.append(name)


def column(data, name, length):
    """A column whether its name uses spaces or underscores (zeros if missing)."""
    for key in (name, name.replace(" ", "_")):
        if key in data:
            return data[key]
    return np.zeros(length)


def cause_shares(data, prefix, causes, window):
    length = len(next(iter(data.values())))
    counts = np.array([column(data, f"{prefix}{c}", length) for c in causes], dtype=float)
    counts = np.array([smooth(c, window) for c in counts])
    total = counts.sum(axis=0)
    total[total == 0] = 1
    return counts / total


# ======================================================================
# training runs (runs/<name>/log.csv)
# ======================================================================
def training_charts(folder, out):
    d = read_csv(os.path.join(folder, "log.csv"))
    x = d["round"]
    window = max(1, len(x) // 60)
    charts = []

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(x, d["avg_days_survived"], color="#3b82c4", alpha=0.25)
    ax.plot(x, smooth(d["avg_days_survived"], window), color="#3b82c4", lw=2.5)
    ax.set(title="How long agents survive", xlabel="training round", ylabel="average days survived")
    save(fig, out, "survival.png", charts)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(x, d["reward_per_day"], color="#2f9e6b", alpha=0.25)
    ax.plot(x, smooth(d["reward_per_day"], window), color="#2f9e6b", lw=2.5)
    ax.axhline(1.0, color="#999", ls="--", lw=1)
    ax.text(x[0], 1.02, "  1.0 = just surviving", color="#777", fontsize=9)
    ax.set(title="Reward per agent per day (higher = better at survival, breeding and comfort)",
           xlabel="training round", ylabel="reward per day")
    save(fig, out, "reward.png", charts)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(x, 100 * d["curiosity"], color="#8a5bd6", lw=2)
    ax.set(title="Curiosity (100% = acting randomly, 0% = always sure)", xlabel="training round",
           ylabel="curiosity %", ylim=(0, 105))
    save(fig, out, "curiosity.png", charts)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(x, smooth(d["births"], window), color="#e84a7f", lw=2.5)
    ax.set(title="Births per training round", xlabel="training round", ylabel="births")
    save(fig, out, "births.png", charts)

    causes = [c for c in DEATH_CAUSES[1:] if np.nansum(column(d, f"died_{c}", len(x))) > 0]
    shares = cause_shares(d, "died_", causes, window * 2)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.stackplot(x, 100 * shares, labels=causes, colors=[CAUSE_COLOURS.get(c, "#ccc") for c in causes], alpha=0.9)
    ax.set(title="What agents die of, over training", xlabel="training round", ylabel="% of deaths",
           ylim=(0, 100))
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9)
    save(fig, out, "causes_of_death.png", charts)
    summary = [f"Training rounds: {int(x[-1])}",
               f"Ticks trained: {int(d['ticks'][-1]):,}",
               f"Days survived: {smooth(d['avg_days_survived'], window)[0]:.1f} at the start, "
               f"{smooth(d['avg_days_survived'], window)[-1]:.1f} now",
               f"Curiosity now: {100 * d['curiosity'][-1]:.0f}%"]
    return charts, summary


# ======================================================================
# history runs (experiments/<name>/)
# ======================================================================
def history_charts(folder, out):
    d = read_csv(os.path.join(folder, "daily.csv"))
    days = d["day"]
    window = max(1, len(days) // 40)
    charts = []

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.stackplot(days, d["men"], d["women"], d["children"], labels=["men", "women", "children"],
                 colors=["#4a7fe0", "#e0506e", "#f2c12e"], alpha=0.9)
    ax.set(title="Population", xlabel="day", ylabel="people alive")
    ax.legend(loc="upper right")
    save(fig, out, "population.png", charts)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(days, smooth(d["births"], window), color="#e84a7f", lw=2.2, label="births")
    ax.plot(days, smooth(d["deaths"], window), color="#555", lw=2.2, label="deaths")
    ax.set(title="Births and deaths per day", xlabel="day", ylabel="per day")
    ax.legend()
    save(fig, out, "births_deaths.png", charts)

    causes = [c for c in DEATH_CAUSES[1:] if np.nansum(column(d, f"died_{c}", len(days))) > 0]
    if causes:
        shares = cause_shares(d, "died_", causes, window * 2)
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.stackplot(days, 100 * shares, labels=causes, colors=[CAUSE_COLOURS.get(c, "#ccc") for c in causes])
        ax.set(title="What people die of, over time", xlabel="day", ylabel="% of deaths", ylim=(0, 100))
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9)
        save(fig, out, "causes_of_death.png", charts)

    fig, ax = plt.subplots(figsize=(11, 5))
    for name in ACTION_NAMES:
        key = f"share_{name.replace(' ', '_')}"
        if key in d:
            ax.plot(days, 100 * smooth(d[key], window), lw=1.8, label=name)
    ax.set(title="What people spend their time doing", xlabel="day", ylabel="% of all actions")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9)
    save(fig, out, "actions.png", charts)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for key, colour in (("attacked", "#e8501c"), ("killed", "#8b0000"), ("hunted", "#7a4b2a"),
                        ("crafted", "#888"), ("planted", "#2f9e6b"), ("harvested", "#d4a017")):
        if key in d:
            ax.plot(days, smooth(d[key], window), lw=2, label=key, color=colour)
    ax.set(title="Violence, hunting, crafting and farming per day", xlabel="day", ylabel="times per day")
    ax.legend()
    save(fig, out, "activities.png", charts)

    if "tribe_0" in d:
        from src.sprites import TRIBE_NAMES, TRIBE_COLOURS
        fig, ax = plt.subplots(figsize=(10, 4.5))
        for t in range(4):
            ax.plot(days, d[f"tribe_{t}"], lw=2.4, label=f"{TRIBE_NAMES[t]} tribe",
                    color=np.array(TRIBE_COLOURS[t]) / 255)
        ax.set(title="Population of each tribe", xlabel="day", ylabel="people alive")
        ax.legend()
        save(fig, out, "tribes.png", charts)

        built = [k for k in d if k.startswith("built_") and np.nansum(d[k]) > 0]
        if built:
            fig, ax = plt.subplots(figsize=(10, 4.5))
            for k in built:
                ax.plot(days, d[k], lw=2, label=k[6:].replace("_", " "))
            ax.set(title="Structures standing in the world", xlabel="day", ylabel="number")
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9)
            save(fig, out, "structures.png", charts)

        armed = [k for k in d if k.startswith("armed_") and np.nansum(d[k]) > 0]
        fig, ax = plt.subplots(figsize=(10, 4.5))
        for k in armed:
            ax.plot(days, d[k], lw=2, label=k[6:].replace("_", " "))
        for k in ("pots", "picks", "sickles"):
            if np.nansum(d[k]) > 0:
                ax.plot(days, d[k], lw=2, ls="--", label=k)
        ax.set(title="Technology: who carries which weapon and tools", xlabel="day", ylabel="people")
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9)
        save(fig, out, "technology.png", charts)

        fig, ax = plt.subplots(figsize=(10, 4.5))
        for key, colour in (("gave", "#2fb39b"), ("stored", "#7a9cc6"), ("called", "#999"),
                            ("caught_sickness", "#7fbf3f"), ("destroyed", "#c0392b")):
            if key in d:
                ax.plot(days, smooth(d[key], window), lw=2, label=key.replace("_", " "), color=colour)
        ax.set(title="Sharing, calls, disease spread and destruction per day", xlabel="day", ylabel="times per day")
        ax.legend()
        save(fig, out, "social.png", charts)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    for ax, g in zip(axes, GENE_NAMES):
        ax.plot(days, d[f"gene_{g}"], color="#8a5bd6", lw=2)
        ax.set(title=f"average {g}", xlabel="day")
    fig.suptitle("Evolution: how the average genes change over generations")
    save(fig, out, "genes.png", charts)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for key, colour in (("rabbits", "#b09070"), ("deer", "#c07a3c"), ("wolves", "#6b6b78"), ("bears", "#5a3a20")):
        ax.plot(days, d[key], lw=2, label=key, color=colour)
    ax.set(title="Animals", xlabel="day", ylabel="alive")
    ax.legend()
    save(fig, out, "animals.png", charts)

    heatmap_chart(folder, out, charts)
    people = family_tree_chart(folder, out, charts)

    last = len(days) - 1
    summary = [f"Days recorded: {int(days[-1])}",
               f"People who ever lived: {people}",
               f"Highest generation reached: {int(np.nanmax(d['max_generation']))}",
               f"Alive at the end: {int(d['population'][last])}",
               f"Total births: {int(np.nansum(d['births']))}, total deaths: {int(np.nansum(d['deaths']))}"]
    return charts, summary


def heatmap_chart(folder, out, charts):
    heat_path = os.path.join(folder, "heatmap.npy")
    if not os.path.exists(heat_path):
        return
    heat = np.load(heat_path).astype(float)
    terrain = np.load(os.path.join(folder, "terrain.npy"))
    base = np.zeros(terrain.shape + (3,))
    for kind in np.unique(terrain):
        base[terrain == kind] = np.array(minimap_colour(kind, False)) / 255
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
    axes[0].imshow(base, interpolation="nearest")
    axes[0].set_title("The world")
    axes[1].imshow(base * 0.45, interpolation="nearest")
    shown = np.ma.masked_where(heat == 0, np.log1p(heat))
    im = axes[1].imshow(shown, cmap="inferno", interpolation="nearest", alpha=0.9)
    axes[1].set_title("Where people spent their time (brighter = more)")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.colorbar(im, ax=axes[1], fraction=0.046, label="time spent (log scale)")
    save(fig, out, "heatmap.png", charts)


def family_tree_chart(folder, out, charts):
    """Draw the family tree of the founder with the most descendants."""
    path = os.path.join(folder, "people.csv")
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        people = list(csv.DictReader(f))
    kids = defaultdict(list)
    for p in people:
        for parent in (p["mother_id"], p["father_id"]):
            if parent not in ("-1", ""):
                kids[parent].append(p["id"])
    by_id = {p["id"]: p for p in people}

    def descendants(pid):
        return sum(1 + descendants(k) for k in kids[pid])

    founders = [p["id"] for p in people if p["founder"] == "1"]
    if not founders:
        return len(people)
    root = max(founders, key=descendants)
    if descendants(root) == 0:
        return len(people)

    # Lay out generations as rows; each person sits above their children
    positions, rows = {}, defaultdict(list)

    def place(pid, depth, left):
        children = kids[pid][:12]
        if depth >= 4 or not children:
            positions[pid] = (left + 0.5, -depth)
            rows[depth].append(pid)
            return left + 1
        x = left
        for c in children:
            x = place(c, depth + 1, x)
        positions[pid] = ((left + x) / 2, -depth)
        rows[depth].append(pid)
        return x

    width = place(root, 0, 0)
    fig, ax = plt.subplots(figsize=(max(8, min(width * 1.3, 30)), 6.5))
    for pid, (x, y) in positions.items():
        for c in kids[pid]:
            if c in positions:
                cx, cy = positions[c]
                ax.plot([x, cx], [y, cy], color="#bbb", lw=1, zorder=1)
    for pid, (x, y) in positions.items():
        p = by_id[pid]
        colour = "#4a7fe0" if p["sex"] == "male" else "#e0506e"
        alive = p["died_tick"] == "-1"
        ax.scatter([x], [y], s=260, color=colour, edgecolor="black" if alive else colour, lw=2, zorder=2)
        cause = "alive" if alive else p["cause"]
        ax.text(x, y - 0.22, f"{p['name']}\n{cause}", ha="center", va="top", fontsize=8)
    ax.set_title(f"Family tree of {by_id[root]['name']} ({descendants(root)} descendants; "
                 "blue = male, red = female, black ring = still alive)")
    ax.axis("off")
    save(fig, out, "family_tree.png", charts)
    return len(people)


def write_report(folder, out, charts, summary, title):
    html = [f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title>",
            "<style>body{font-family:Segoe UI,sans-serif;max-width:1200px;margin:30px auto;padding:0 16px;"
            "background:#fbfaf7;color:#1f1d24}img{width:100%;border:1px solid #ddd;border-radius:8px;"
            "margin:10px 0 26px;background:white}li{margin:4px 0}</style></head><body>",
            f"<h1>{title}</h1><ul>" + "".join(f"<li>{s}</li>" for s in summary) + "</ul>"]
    html += [f"<img src='{c}' alt='{c}'>" for c in charts]
    html.append("</body></html>")
    path = os.path.join(out, "report.html")
    with open(path, "w") as f:
        f.write("\n".join(html))
    return path


def main():
    parser = argparse.ArgumentParser(description="Make charts from training or a history run")
    parser.add_argument("folder", help="runs/<name> or experiments/<name>")
    folder = parser.parse_args().folder
    out = os.path.join(folder, "charts")
    os.makedirs(out, exist_ok=True)
    if os.path.exists(os.path.join(folder, "daily.csv")):
        charts, summary = history_charts(folder, out)
        title = f"History run: {os.path.basename(folder.rstrip('/'))}"
    elif os.path.exists(os.path.join(folder, "log.csv")):
        charts, summary = training_charts(folder, out)
        title = f"Training run: {os.path.basename(folder.rstrip('/'))}"
    else:
        raise SystemExit(f"No log.csv or daily.csv in {folder}")
    report = write_report(folder, out, charts, summary, title)
    print("\n".join(summary))
    print(f"\n{len(charts)} charts saved in {out}")
    print(f"Open this in your browser: {report}")


if __name__ == "__main__":
    main()
