"""Rummikub tournament data analysis.

Reads matchup_results.csv and adaptive_weights.json to produce:
  - Overall agent rankings (win rate, games, avg hand size, avg turns)
  - Head-to-head win rate matrix
  - Per-run breakdowns showing consistency across runs
  - ELO ratings computed from game history
  - Adaptive agent weight evolution summary
  - Optional matplotlib charts (heatmaps, trends, ELO bar chart)

Usage:
    py analyze_results.py                  # text reports only
    py analyze_results.py --plots          # text + save PNG charts
    py analyze_results.py --plots --show   # text + display charts interactively
"""

import csv
import json
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(_SCRIPT_DIR, "matchup_results.csv")
DEFAULT_WEIGHTS = os.path.join(_SCRIPT_DIR, "adaptive_weights.json")
PLOTS_DIR = os.path.join(_SCRIPT_DIR, "plots")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_games(csv_path: str = DEFAULT_CSV) -> List[dict]:
    """Load all game rows from the matchup CSV."""
    if not os.path.exists(csv_path):
        print(f"Error: CSV not found at {csv_path}")
        sys.exit(1)
    with open(csv_path, "r", newline="") as f:
        return list(csv.DictReader(f))


def load_adaptive_weights(json_path: str = DEFAULT_WEIGHTS) -> Optional[dict]:
    """Load adaptive agent weight history if available."""
    if not os.path.exists(json_path):
        return None
    with open(json_path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Text reports
# ---------------------------------------------------------------------------


def agent_rankings(games: List[dict]) -> List[dict]:
    """Compute per-agent stats sorted by win rate descending."""
    stats: Dict[str, dict] = {}
    for g in games:
        for side in ("agent1", "agent2"):
            name = g[side]
            if name not in stats:
                stats[name] = {
                    "name": name,
                    "wins": 0,
                    "games": 0,
                    "total_hand": 0,
                    "total_turns": 0,
                    "decisive_wins": 0,  # wins where hand_size == 0
                }
            stats[name]["games"] += 1
            hand_key = f"{side}_hand_size"
            hand = int(g[hand_key]) if g[hand_key] not in ("", "-1") else -1
            if hand >= 0:
                stats[name]["total_hand"] += hand
            stats[name]["total_turns"] += int(g["num_turns"])
            if g["winner"] == name:
                stats[name]["wins"] += 1
                if hand == 0:
                    stats[name]["decisive_wins"] += 1

    rankings = sorted(
        stats.values(), key=lambda s: s["wins"] / max(s["games"], 1), reverse=True
    )
    return rankings


def print_rankings(rankings: List[dict]):
    """Print formatted agent rankings table."""
    print("\n" + "=" * 90)
    print(f"{'AGENT RANKINGS':^90}")
    print("=" * 90)
    hdr = (
        f"{'Rank':<6}{'Agent':<15}{'Wins':<7}{'Games':<8}{'Win %':<9}"
        f"{'Avg Hand':<10}{'Avg Turns':<11}{'Clean Wins':<12}"
    )
    print(hdr)
    print("-" * 90)
    for i, s in enumerate(rankings, 1):
        wr = s["wins"] / max(s["games"], 1) * 100
        avg_hand = s["total_hand"] / max(s["games"], 1)
        avg_turns = s["total_turns"] / max(s["games"], 1)
        clean_pct = s["decisive_wins"] / max(s["wins"], 1) * 100
        print(
            f"{i:<6}{s['name']:<15}{s['wins']:<7}{s['games']:<8}"
            f"{wr:>5.1f}%   {avg_hand:>6.1f}   {avg_turns:>7.0f}     "
            f"{s['decisive_wins']}/{s['wins']} ({clean_pct:.0f}%)"
        )
    print()


def head_to_head(games: List[dict]) -> Tuple[List[str], Dict[str, Dict[str, dict]]]:
    """Build head-to-head stats: h2h[a][b] = {wins, losses, games}."""
    agents = sorted({g["agent1"] for g in games} | {g["agent2"] for g in games})
    h2h: Dict[str, Dict[str, dict]] = {
        a: {b: {"wins": 0, "losses": 0, "games": 0} for b in agents} for a in agents
    }
    for g in games:
        a1, a2, winner = g["agent1"], g["agent2"], g["winner"]
        h2h[a1][a2]["games"] += 1
        h2h[a2][a1]["games"] += 1
        if winner == a1:
            h2h[a1][a2]["wins"] += 1
            h2h[a2][a1]["losses"] += 1
        elif winner == a2:
            h2h[a2][a1]["wins"] += 1
            h2h[a1][a2]["losses"] += 1

    return agents, h2h


def print_h2h_matrix(agents: List[str], h2h: Dict[str, Dict[str, dict]]):
    """Print head-to-head win rate matrix."""
    total_games = sum(h2h[a][b]["games"] for a in agents for b in agents) // 2
    num_runs = len(
        {g for a in agents for b in agents for g in range(h2h[a][b]["games"])}
    )  # approximate

    print("=" * 90)
    print(f"{'HEAD-TO-HEAD WIN RATES':^90}")
    print("=" * 90)

    col_w = 11
    header = f"{'':>{col_w}}"
    for a in agents:
        label = a[:col_w]
        header += f" {label:>{col_w}}"
    print(header)
    print("-" * len(header))

    for row_agent in agents:
        line = f"{row_agent:>{col_w}}"
        for col_agent in agents:
            if row_agent == col_agent:
                line += f" {'---':>{col_w}}"
            else:
                s = h2h[row_agent][col_agent]
                if s["games"] > 0:
                    wr = s["wins"] / s["games"] * 100
                    g = s["games"]
                    cell = f"{wr:.0f}%({g})"
                    line += f" {cell:>{col_w}}"
                else:
                    line += f" {'n/a':>{col_w}}"
        print(line)

    print()
    print("(Row = agent, Column = opponent, Value = row agent's win% (games played))")
    print()


def per_run_breakdown(games: List[dict]):
    """Show win rates per run to assess consistency."""
    runs: Dict[str, List[dict]] = defaultdict(list)
    for g in games:
        runs[g["run_id"]].append(g)

    print("=" * 90)
    print(f"{'PER-RUN BREAKDOWN':^90}")
    print("=" * 90)

    agents = sorted({g["agent1"] for g in games} | {g["agent2"] for g in games})
    col_w = 12

    header = f"{'Run ID':<20}{'Games':<8}"
    for a in agents:
        header += f"{a[:col_w]:>{col_w}}"
    print(header)
    print("-" * len(header))

    for run_id in sorted(runs.keys()):
        run_games = runs[run_id]
        wins = defaultdict(int)
        played = defaultdict(int)
        for g in run_games:
            played[g["agent1"]] += 1
            played[g["agent2"]] += 1
            if g["winner"]:
                wins[g["winner"]] += 1

        line = f"{run_id:<20}{len(run_games):<8}"
        for a in agents:
            if played[a] > 0:
                wr = wins[a] / played[a] * 100
                cell = f"{wr:.0f}%({played[a]})"
                line += f"{cell:>{col_w}}"
            else:
                line += f"{'n/a':>{col_w}}"
        print(line)

    # Consistency metric: standard deviation of win rates across runs
    print()
    print("Consistency (std dev of win% across runs):")
    for a in agents:
        run_wrs = []
        for run_id, run_games in runs.items():
            played_count = sum(
                1 for g in run_games if g["agent1"] == a or g["agent2"] == a
            )
            win_count = sum(1 for g in run_games if g["winner"] == a)
            if played_count >= 3:  # need minimum sample
                run_wrs.append(win_count / played_count * 100)
        if len(run_wrs) >= 2:
            mean = sum(run_wrs) / len(run_wrs)
            variance = sum((x - mean) ** 2 for x in run_wrs) / len(run_wrs)
            std = math.sqrt(variance)
            print(
                f"  {a:<15} std = {std:>5.1f}%  (across {len(run_wrs)} runs, mean {mean:.1f}%)"
            )
        else:
            print(f"  {a:<15} insufficient data ({len(run_wrs)} qualifying runs)")
    print()


# ---------------------------------------------------------------------------
# ELO ratings
# ---------------------------------------------------------------------------


def compute_elo(
    games: List[dict], k: float = 32.0, initial: float = 1500.0
) -> Dict[str, float]:
    """Compute ELO ratings from chronological game results."""
    elo: Dict[str, float] = defaultdict(lambda: initial)

    for g in games:
        a1, a2, winner = g["agent1"], g["agent2"], g["winner"]
        r1, r2 = elo[a1], elo[a2]

        # Expected scores
        e1 = 1.0 / (1.0 + 10.0 ** ((r2 - r1) / 400.0))
        e2 = 1.0 - e1

        # Actual scores
        if winner == a1:
            s1, s2 = 1.0, 0.0
        elif winner == a2:
            s1, s2 = 0.0, 1.0
        else:
            s1, s2 = 0.5, 0.5  # draw / timeout

        elo[a1] = r1 + k * (s1 - e1)
        elo[a2] = r2 + k * (s2 - e2)

    return dict(elo)


def print_elo(elo: Dict[str, float]):
    """Print ELO ratings sorted descending."""
    print("=" * 50)
    print(f"{'ELO RATINGS':^50}")
    print("=" * 50)
    print(f"{'Rank':<6}{'Agent':<20}{'ELO':<10}{'vs 1500'}")
    print("-" * 50)
    sorted_elo = sorted(elo.items(), key=lambda x: x[1], reverse=True)
    for i, (agent, rating) in enumerate(sorted_elo, 1):
        diff = rating - 1500
        sign = "+" if diff >= 0 else ""
        print(f"{i:<6}{agent:<20}{rating:>7.0f}   {sign}{diff:.0f}")
    print()


# ---------------------------------------------------------------------------
# Adaptive weight analysis
# ---------------------------------------------------------------------------


def print_adaptive_analysis(data: dict):
    """Print adaptive agent weight evolution summary."""
    print("=" * 70)
    print(f"{'ADAPTIVE AGENT WEIGHT EVOLUTION':^70}")
    print("=" * 70)

    stats = data.get("stats", {})
    print(f"Total games: {stats.get('games_played', '?')}")
    print(
        f"Wins: {stats.get('wins', '?')} "
        f"({stats.get('wins', 0) / max(stats.get('games_played', 1), 1) * 100:.1f}%)"
    )
    print(f"Best win streak: {data.get('best_win_streak', '?')}")
    print()

    # Current vs best weights
    current = data.get("weights", {})
    best = data.get("best_weights", {})
    print(f"{'Weight':<25}{'Current':>10}{'Best':>10}{'Delta':>10}")
    print("-" * 55)
    for key in ("aggressiveness", "meld_size_preference", "draw_preference"):
        c = current.get(key, 0)
        b = best.get(key, 0)
        delta = c - b
        sign = "+" if delta >= 0 else ""
        print(f"  {key:<23}{c:>9.3f} {b:>9.3f} {sign}{delta:>8.3f}")
    print()

    # Weight trajectory summary (first, middle, last)
    history = data.get("weight_history", [])
    if history:
        n = len(history)
        checkpoints = [0, n // 4, n // 2, 3 * n // 4, n - 1]
        checkpoints = sorted(set(checkpoints))  # deduplicate for small histories
        print(f"{'Game':<8}{'Won':<6}{'LR':<8}{'Agg':<8}{'Meld':<8}{'Draw':<8}")
        print("-" * 46)
        for idx in checkpoints:
            h = history[idx]
            won = "W" if h["won"] else "L"
            print(
                f"{h['game']:<8}{won:<6}{h['lr']:<8.4f}"
                f"{h['aggressiveness']:<8.3f}"
                f"{h['meld_size_preference']:<8.3f}"
                f"{h['draw_preference']:<8.3f}"
            )

        # Win rate in windows
        print()
        window = max(10, n // 5)
        print(f"Win rate over time (window = {window} games):")
        for start in range(0, n, window):
            chunk = history[start : start + window]
            wins = sum(1 for h in chunk if h["won"])
            wr = wins / len(chunk) * 100
            bar_len = int(wr / 2)
            bar = "#" * bar_len + "." * (50 - bar_len)
            print(
                f"  Games {start + 1:>4}-{start + len(chunk):>4}: "
                f"{bar} {wr:>5.1f}% ({wins}/{len(chunk)})"
            )
    print()


# ---------------------------------------------------------------------------
# Game duration analysis
# ---------------------------------------------------------------------------


def print_game_duration_stats(games: List[dict]):
    """Analyze game duration (num_turns) patterns."""
    print("=" * 70)
    print(f"{'GAME DURATION ANALYSIS':^70}")
    print("=" * 70)

    turns = [int(g["num_turns"]) for g in games]
    max_turns = max(turns)
    min_turns = min(turns)
    avg_turns = sum(turns) / len(turns)
    median_turns = sorted(turns)[len(turns) // 2]
    timeouts = sum(1 for t in turns if t >= 500)

    print(f"Total games analyzed: {len(games)}")
    print(
        f"Turn count: min={min_turns}, max={max_turns}, avg={avg_turns:.0f}, median={median_turns}"
    )
    print(
        f"Timeouts (>=500 turns): {timeouts}/{len(games)} ({timeouts / len(games) * 100:.1f}%)"
    )
    print()

    # Timeout rate per agent
    agents = sorted({g["agent1"] for g in games} | {g["agent2"] for g in games})
    print(f"{'Agent':<15}{'Timeout Rate':<15}{'Avg Turns':<12}{'Decisive %':<12}")
    print("-" * 54)
    for a in agents:
        agent_games = [g for g in games if g["agent1"] == a or g["agent2"] == a]
        agent_turns = [int(g["num_turns"]) for g in agent_games]
        agent_timeouts = sum(1 for t in agent_turns if t >= 500)
        decisive = sum(
            1 for g in agent_games if int(g["num_turns"]) < 500 and g["winner"] == a
        )
        agent_wins = sum(1 for g in agent_games if g["winner"] == a)
        decisive_pct = decisive / max(agent_wins, 1) * 100
        to_rate = agent_timeouts / max(len(agent_games), 1) * 100
        avg_t = sum(agent_turns) / max(len(agent_turns), 1)
        print(
            f"  {a:<13}{to_rate:>5.1f}%       {avg_t:>7.0f}     {decisive_pct:>5.1f}%"
        )

    # Distribution histogram
    print()
    print("Turn distribution:")
    buckets = [(0, 100), (100, 200), (200, 300), (300, 400), (400, 499), (500, 501)]
    bucket_labels = ["1-100", "101-200", "201-300", "301-400", "401-499", "500 (TO)"]
    for (lo, hi), label in zip(buckets, bucket_labels):
        count = (
            sum(1 for t in turns if lo < t <= hi)
            if lo > 0
            else sum(1 for t in turns if lo <= t <= hi)
        )
        # Fix: include lower bound properly
        count = sum(1 for t in turns if lo <= t < hi)
        if label == "500 (TO)":
            count = sum(1 for t in turns if t >= 500)
        pct = count / len(turns) * 100
        bar_len = int(pct)
        bar = "#" * bar_len
        print(f"  {label:<10} {bar:<40} {count:>4} ({pct:.1f}%)")
    print()


# ---------------------------------------------------------------------------
# Matplotlib visualizations (optional)
# ---------------------------------------------------------------------------


def _try_import_matplotlib():
    """Try to import matplotlib, return (plt, sns_available) or (None, False)."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt

        try:
            import seaborn as sns

            return plt, True
        except ImportError:
            return plt, False
    except ImportError:
        return None, False


def plot_h2h_heatmap(
    agents: List[str], h2h: Dict[str, Dict[str, dict]], show: bool = False
):
    """Save head-to-head win rate heatmap as PNG."""
    plt, has_sns = _try_import_matplotlib()
    if plt is None:
        print("  [skip] matplotlib not installed — no heatmap")
        return

    import numpy as np

    n = len(agents)
    matrix = np.full((n, n), float("nan"))
    for i, a in enumerate(agents):
        for j, b in enumerate(agents):
            if i != j and h2h[a][b]["games"] > 0:
                matrix[i][j] = h2h[a][b]["wins"] / h2h[a][b]["games"] * 100

    fig, ax = plt.subplots(figsize=(8, 6))

    if has_sns:
        import seaborn as sns

        sns.heatmap(
            matrix,
            annot=True,
            fmt=".0f",
            xticklabels=agents,
            yticklabels=agents,
            cmap="RdYlGn",
            center=50,
            vmin=0,
            vmax=100,
            ax=ax,
            mask=np.isnan(matrix),
            linewidths=0.5,
            cbar_kws={"label": "Win %"},
        )
    else:
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=100)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(agents, rotation=45, ha="right")
        ax.set_yticklabels(agents)
        plt.colorbar(im, ax=ax, label="Win %")
        for i in range(n):
            for j in range(n):
                if not np.isnan(matrix[i][j]):
                    ax.text(
                        j,
                        i,
                        f"{matrix[i][j]:.0f}",
                        ha="center",
                        va="center",
                        fontsize=9,
                    )

    ax.set_title("Head-to-Head Win Rates (%)")
    ax.set_xlabel("Opponent")
    ax.set_ylabel("Agent")
    plt.tight_layout()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, "h2h_heatmap.png")
    fig.savefig(path, dpi=150)
    print(f"  Saved: {path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_elo_ratings(elo: Dict[str, float], show: bool = False):
    """Save ELO rating bar chart as PNG."""
    plt, has_sns = _try_import_matplotlib()
    if plt is None:
        print("  [skip] matplotlib not installed — no ELO chart")
        return

    sorted_agents = sorted(elo.items(), key=lambda x: x[1], reverse=True)
    names = [a for a, _ in sorted_agents]
    ratings = [r for _, r in sorted_agents]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2ecc71" if r >= 1500 else "#e74c3c" for r in ratings]
    bars = ax.barh(names[::-1], ratings[::-1], color=colors[::-1], edgecolor="white")

    ax.axvline(x=1500, color="gray", linestyle="--", alpha=0.7, label="Baseline (1500)")
    ax.set_xlabel("ELO Rating")
    ax.set_title("Agent ELO Ratings")
    ax.legend()

    for bar, rating in zip(bars, ratings[::-1]):
        ax.text(
            bar.get_width() + 5,
            bar.get_y() + bar.get_height() / 2,
            f"{rating:.0f}",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, "elo_ratings.png")
    fig.savefig(path, dpi=150)
    print(f"  Saved: {path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_adaptive_evolution(data: dict, show: bool = False):
    """Save adaptive weight evolution chart as PNG."""
    plt, _ = _try_import_matplotlib()
    if plt is None:
        print("  [skip] matplotlib not installed — no evolution chart")
        return

    history = data.get("weight_history", [])
    if not history:
        print("  [skip] no weight history data")
        return

    game_nums = [h["game"] for h in history]
    agg = [h["aggressiveness"] for h in history]
    meld = [h["meld_size_preference"] for h in history]
    draw = [h["draw_preference"] for h in history]
    lr = [h["lr"] for h in history]
    wins = [h["won"] for h in history]

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # Panel 1: Weights over time
    ax1 = axes[0]
    ax1.plot(game_nums, agg, label="Aggressiveness", color="#e74c3c", linewidth=1.5)
    ax1.plot(game_nums, meld, label="Meld Size Pref", color="#3498db", linewidth=1.5)
    ax1.plot(game_nums, draw, label="Draw Pref", color="#2ecc71", linewidth=1.5)
    ax1.set_ylabel("Weight Value")
    ax1.set_title("Adaptive Agent — Weight Evolution")
    ax1.legend(loc="upper right")
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Win/loss markers + rolling win rate
    ax2 = axes[1]
    win_games = [g for g, w in zip(game_nums, wins) if w]
    loss_games = [g for g, w in zip(game_nums, wins) if not w]
    ax2.scatter(
        win_games,
        [1] * len(win_games),
        color="#2ecc71",
        marker="|",
        s=30,
        alpha=0.6,
        label="Win",
    )
    ax2.scatter(
        loss_games,
        [0] * len(loss_games),
        color="#e74c3c",
        marker="|",
        s=30,
        alpha=0.6,
        label="Loss",
    )

    # Rolling win rate
    window = max(10, len(history) // 10)
    rolling_wr = []
    for i in range(len(wins)):
        start = max(0, i - window + 1)
        chunk = wins[start : i + 1]
        rolling_wr.append(sum(chunk) / len(chunk))
    ax2.plot(
        game_nums,
        rolling_wr,
        color="#f39c12",
        linewidth=2,
        label=f"Win rate (rolling {window})",
    )
    ax2.set_ylabel("Win Rate")
    ax2.set_ylim(-0.1, 1.1)
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    # Panel 3: Learning rate decay
    ax3 = axes[2]
    ax3.plot(game_nums, lr, color="#9b59b6", linewidth=1.5)
    ax3.set_ylabel("Learning Rate")
    ax3.set_xlabel("Game Number")
    ax3.set_title("Learning Rate Decay")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, "adaptive_evolution.png")
    fig.savefig(path, dpi=150)
    print(f"  Saved: {path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_win_rate_over_time(games: List[dict], show: bool = False):
    """Plot cumulative win rate for each agent over the full game history."""
    plt, _ = _try_import_matplotlib()
    if plt is None:
        print("  [skip] matplotlib not installed — no win rate trend chart")
        return

    agents = sorted({g["agent1"] for g in games} | {g["agent2"] for g in games})
    cum_wins = {a: [] for a in agents}
    cum_games = {a: [] for a in agents}
    totals = {a: {"w": 0, "g": 0} for a in agents}

    for g in games:
        a1, a2, winner = g["agent1"], g["agent2"], g["winner"]
        for a in (a1, a2):
            totals[a]["g"] += 1
            if winner == a:
                totals[a]["w"] += 1
            cum_wins[a].append(totals[a]["w"])
            cum_games[a].append(totals[a]["g"])

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [
        "#e74c3c",
        "#3498db",
        "#2ecc71",
        "#f39c12",
        "#9b59b6",
        "#1abc9c",
        "#e67e22",
    ]
    for i, agent in enumerate(agents):
        if cum_games[agent]:
            wr = [w / g * 100 for w, g in zip(cum_wins[agent], cum_games[agent])]
            ax.plot(
                range(1, len(wr) + 1),
                wr,
                label=agent,
                color=colors[i % len(colors)],
                linewidth=1.5,
                alpha=0.85,
            )

    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Games Played (per agent)")
    ax.set_ylabel("Cumulative Win Rate (%)")
    ax.set_title("Win Rate Convergence Over Time")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    plt.tight_layout()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, "win_rate_over_time.png")
    fig.savefig(path, dpi=150)
    print(f"  Saved: {path}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    do_plots = "--plots" in sys.argv
    show_plots = "--show" in sys.argv

    print(f"\nLoading data from: {DEFAULT_CSV}")
    games = load_games()
    adaptive = load_adaptive_weights()
    print(
        f"Loaded {len(games)} games across "
        f"{len(set(g['run_id'] for g in games))} run(s)\n"
    )

    # --- Text reports (always) ---
    rankings = agent_rankings(games)
    print_rankings(rankings)

    agents, h2h = head_to_head(games)
    print_h2h_matrix(agents, h2h)

    per_run_breakdown(games)

    elo = compute_elo(games)
    print_elo(elo)

    print_game_duration_stats(games)

    if adaptive:
        print_adaptive_analysis(adaptive)
    else:
        print("(No adaptive_weights.json found — skipping weight analysis)\n")

    # --- Plots (optional) ---
    if do_plots:
        print("=" * 50)
        print("GENERATING PLOTS")
        print("=" * 50)
        plot_h2h_heatmap(agents, h2h, show=show_plots)
        plot_elo_ratings(elo, show=show_plots)
        plot_win_rate_over_time(games, show=show_plots)
        if adaptive:
            plot_adaptive_evolution(adaptive, show=show_plots)
        print()
    elif not do_plots:
        print("Tip: run with --plots to generate PNG charts, --show to display them.\n")


if __name__ == "__main__":
    main()
