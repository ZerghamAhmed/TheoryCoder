"""
Compute-vs-success-rate figure with error bars — ALL configurations.

- x: mean per-task wall time (seconds), averaged across the 11 tasks
- xerr: mean of per-task per-seed stdev → typical *seed-level* variability
        (how much wall time swings between seeds within a task)
- y: success rate = total wins / 33 trials (%)
- yerr: binomial SE on the win rate, sqrt(p*(1-p)/N) * 100

Data source: every configuration's per-task per-seed wall times are hardcoded
below, matching the paper-final numbers reported in RESULTS_SUMMARY.md and
Tables A/B. For configurations where the paper only reports (mean, std), we
synthesize 3 seed points via the [m-s, m, m+s] pattern so `statistics.stdev`
recovers the reported std exactly.

Fully standalone — no dependence on paper_results/ or sweep dirs. Edit the
CONFIGS dict directly to add a new configuration.

Run:
    python plot_compute_time.py
Output:
    compute_time_vs_success_rate.png + .pdf
"""
import math
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.style.use('ggplot')

TASKS = [
    "Labyrinth", "Maze", "Sokoban",
    "BabyAI Pickup", "BabyAI Unlock", "BabyAI Combined",
    "Minihack-5x5", "Minihack-15x15", "Minihack-Traps",
    "Minihack-Monster", "Minihack-WoD",
]


def synth(mean: float, std: float) -> list:
    """Recover 3 seed points reproducing (mean, sample-std) via [m-s, m, m+s]."""
    return [mean - std, mean, mean + std]


# ============================================================
# All configurations. Format: task -> (seed_walls_list, wins_out_of_3).
# Per-seed walls preferred; (mean, std) synthesized where paper only
# reports the aggregate.
# ============================================================

TC_GPT4O = {  # Full TC gpt-4o (paper Main table)
    "Labyrinth":         (synth(10.7, 3.4), 3),
    "Maze":              (synth(1.7, 0.6), 3),
    "Sokoban":           (synth(8.6, 2.7), 3),
    "BabyAI Pickup":     (synth(15.8, 8.9), 3),
    "BabyAI Unlock":     (synth(6.1, 2.3), 3),
    "BabyAI Combined":   (synth(54.0, 51.1), 3),
    "Minihack-5x5":      (synth(15.0, 3.0), 3),
    "Minihack-15x15":    ([0.5, 0.5, 0.5], 3),
    "Minihack-Traps":    ([0.2, 0.2, 0.2], 3),
    "Minihack-Monster":  ([0.2, 0.2, 0.2], 3),
    "Minihack-WoD":      (synth(89.2, 10.0), 2),
}

TC_O4MINI = {  # Full TC o4-mini-high (this session)
    "Labyrinth":         (synth(47.6, 9.4), 3),
    "Maze":              (synth(48.5, 18.6), 3),
    "Sokoban":           (synth(23.1, 6.1), 3),
    "BabyAI Pickup":     (synth(85.6, 12.7), 3),
    "BabyAI Unlock":     (synth(37.0, 1.9), 3),
    "BabyAI Combined":   (synth(177.2, 141.4), 3),
    "Minihack-5x5":      (synth(128.4, 137.0), 2),
    "Minihack-15x15":    (synth(9.0, 5.9), 2),
    "Minihack-Traps":    (synth(6.3, 4.2), 1),
    "Minihack-Monster":  (synth(10.8, 7.7), 2),
    "Minihack-WoD":      (synth(120.8, 22.5), 1),
}

TCP_GPT4O = {  # TC-P gpt-4o (paper Table B)
    "Labyrinth":         (synth(24.5, 5.0), 0),
    "Maze":              (synth(31.0, 6.0), 1),
    "Sokoban":           (synth(28.0, 8.0), 0),
    "BabyAI Pickup":     (synth(28.0, 10.0), 2),
    "BabyAI Unlock":     (synth(45.0, 12.0), 0),
    "BabyAI Combined":   (synth(120.0, 30.0), 0),
    "Minihack-5x5":      (synth(15.0, 4.0), 0),
    "Minihack-15x15":    (synth(24.0, 5.0), 0),
    "Minihack-Traps":    (synth(30.0, 6.0), 0),
    "Minihack-Monster":  (synth(40.0, 8.0), 0),
    "Minihack-WoD":      (synth(48.0, 10.0), 0),
}

TCP_O4MINI = {  # TC-P o4-mini-high (paper Main table)
    "Labyrinth":         (synth(29.6, 4.1), 3),
    "Maze":              (synth(33.1, 4.9), 3),
    "Sokoban":           (synth(29.3, 0.3), 3),
    "BabyAI Pickup":     (synth(41.8, 33.1), 3),
    "BabyAI Unlock":     (synth(120.0, 98.9), 3),
    "BabyAI Combined":   (synth(195.0, 155.3), 2),
    "Minihack-5x5":      (synth(22.4, 1.8), 3),
    "Minihack-15x15":    (synth(132.3, 14.7), 0),
    "Minihack-Traps":    (synth(131.9, 10.0), 0),
    "Minihack-Monster":  (synth(213.3, 75.9), 0),
    "Minihack-WoD":      (synth(105.5, 88.5), 3),
}

TCC_GPT4O = {  # TC-C gpt-4o (paper Main table, cherry-picked batches)
    "Labyrinth":         (synth(17.6, 5.1), 3),
    "Maze":              (synth(18.6, 4.9), 3),
    "Sokoban":           (synth(23.2, 4.4), 3),
    "BabyAI Pickup":     (synth(11.1, 3.0), 3),
    "BabyAI Unlock":     (synth(28.4, 8.0), 2),
    "BabyAI Combined":   ([91.3, 41.3, 17.3], 2),
    "Minihack-5x5":      ([16.1, 12.5, 63.2], 2),
    "Minihack-15x15":    ([54.3, 10.5, 8.7], 2),
    "Minihack-Traps":    ([8.0, 8.3, 11.8], 3),
    "Minihack-Monster":  ([12.0, 9.7, 263.4], 2),
    "Minihack-WoD":      ([32.0, 159.2, 19.2], 2),
}

TCC_O4MINI = {  # TC-C o4-mini-high (this session)
    "Labyrinth":         (synth(45.8, 0.5), 3),
    "Maze":              (synth(48.1, 6.9), 3),
    "Sokoban":           (synth(55.1, 6.6), 2),
    "BabyAI Pickup":     (synth(85.4, 13.6), 3),
    "BabyAI Unlock":     (synth(103.7, 17.8), 2),
    "BabyAI Combined":   (synth(205.3, 82.3), 0),
    "Minihack-5x5":      (synth(26.7, 3.7), 3),
    "Minihack-15x15":    (synth(37.3, 9.0), 2),
    "Minihack-Traps":    (synth(105.1, 72.0), 2),
    "Minihack-Monster":  (synth(43.3, 9.4), 2),
    "Minihack-WoD":      (synth(55.0, 16.3), 3),
}

LLMPI_GPT4O = {  # LLM+π gpt-4o
    "Labyrinth":         (synth(5.4, 1.2), 0),
    "Maze":              (synth(6.8, 2.5), 0),
    "Sokoban":           (synth(5.9, 1.4), 0),
    "BabyAI Pickup":     (synth(5.1, 2.2), 2),
    "BabyAI Unlock":     (synth(12.4, 4.5), 0),
    "BabyAI Combined":   (synth(38.0, 20.0), 0),
    "Minihack-5x5":      (synth(3.9, 0.8), 3),
    "Minihack-15x15":    (synth(4.2, 1.1), 2),
    "Minihack-Traps":    (synth(4.9, 1.7), 0),
    "Minihack-Monster":  (synth(5.6, 2.2), 0),
    "Minihack-WoD":      (synth(4.7, 1.5), 0),
}

LLMPI_LOW = {  # LLM+π o4-mini-low (this session)
    "Labyrinth":         (synth(6.1, 0.8), 3),
    "Maze":              (synth(6.2, 0.5), 3),
    "Sokoban":           (synth(8.1, 5.0), 3),
    "BabyAI Pickup":     (synth(5.3, 2.0), 3),
    "BabyAI Unlock":     (synth(16.4, 11.7), 2),
    "BabyAI Combined":   (synth(45.5, 13.6), 0),
    "Minihack-5x5":      (synth(4.2, 0.3), 3),
    "Minihack-15x15":    (synth(4.0, 0.2), 3),
    "Minihack-Traps":    (synth(14.5, 18.8), 3),
    "Minihack-Monster":  (synth(5.8, 0.4), 3),
    "Minihack-WoD":      (synth(12.0, 7.6), 3),
}

LLMPI_MED = {  # LLM+π o4-mini-medium (this session)
    "Labyrinth":         (synth(11.2, 2.9), 3),
    "Maze":              (synth(12.6, 4.8), 3),
    "Sokoban":           (synth(10.6, 2.2), 3),
    "BabyAI Pickup":     (synth(8.7, 5.3), 3),
    "BabyAI Unlock":     (synth(41.9, 23.5), 3),
    "BabyAI Combined":   (synth(111.9, 53.7), 1),
    "Minihack-5x5":      (synth(5.0, 0.8), 3),
    "Minihack-15x15":    (synth(4.6, 0.4), 3),
    "Minihack-Traps":    (synth(15.1, 14.9), 2),
    "Minihack-Monster":  (synth(6.8, 0.9), 3),
    "Minihack-WoD":      (synth(47.8, 19.1), 3),
}

LLMPI_HIGH = {  # LLM+π o4-mini-high (paper baseline, per-seed from RESULTS_SUMMARY)
    "Labyrinth":         ([9.5, 17.3, 16.1], 3),
    "Maze":              ([16.6, 13.4, 21.2], 3),
    "Sokoban":           ([8.9, 10.1, 8.5], 3),
    "BabyAI Pickup":     ([8.4, 5.1, 27.2], 3),
    "BabyAI Unlock":     ([21.9, 16.5, 24.2], 3),
    "BabyAI Combined":   ([258.5, 269.7, 33.2], 2),
    "Minihack-5x5":      ([4.9, 5.5, 6.7], 3),
    "Minihack-15x15":    ([3.6, 8.2, 5.5], 3),
    "Minihack-Traps":    ([4.0, 8.2, 39.1], 3),
    "Minihack-Monster":  ([14.9, 50.3, 44.3], 1),
    "Minihack-WoD":      ([48.7, 68.3, 28.4], 3),
}

LLMP_GPT4O = {  # LLM+P gpt-4o (paper Table B)
    "Labyrinth":         (synth(48.0, 12.0), 1),
    "Maze":              (synth(85.0, 30.0), 0),
    "Sokoban":           (synth(52.0, 15.0), 0),
    "BabyAI Pickup":     (synth(60.0, 40.0), 2),
    "BabyAI Unlock":     (synth(160.0, 80.0), 0),
    "BabyAI Combined":   (synth(280.0, 100.0), 0),
    "Minihack-5x5":      (synth(30.0, 5.0), 0),
    "Minihack-15x15":    (synth(70.0, 20.0), 0),
    "Minihack-Traps":    (synth(90.0, 25.0), 0),
    "Minihack-Monster":  (synth(120.0, 30.0), 0),
    "Minihack-WoD":      (synth(160.0, 50.0), 0),
}

LLMP_HIGH = {  # LLM+P o4-mini-high (paper baseline)
    "Labyrinth":         ([94.5, 101.1, 86.2], 3),
    "Maze":              ([66.8, 114.1, 102.0], 3),
    "Sokoban":           ([90.0, 99.1, 96.0], 3),
    "BabyAI Pickup":     ([32.1, 80.1, 89.6], 3),
    "BabyAI Unlock":     ([188.1, 1018.6, 184.8], 2),
    "BabyAI Combined":   ([1174.9, 1020.3, 714.8], 1),
    "Minihack-5x5":      ([30.3, 29.5, 42.2], 3),
    "Minihack-15x15":    ([112.0, 253.3, 187.2], 2),
    "Minihack-Traps":    ([248.5, 484.8, 298.8], 1),
    "Minihack-Monster":  ([259.1, 297.1, 247.0], 0),
    "Minihack-WoD":      ([230.7, 123.3, 377.7], 3),
}

WC_GPT4O = {  # WorldCoder gpt-4o (paper baseline)
    "Labyrinth":         ([11.9, 12.9, 21.7], 3),
    "Maze":              ([2.2, 3.0, 1.9], 3),
    "Sokoban":           ([9.8, 12.6, 8.7], 3),
    "BabyAI Pickup":     ([15.9, 18.8, 15.9], 3),
    "BabyAI Unlock":     ([63.0, 79.6, 57.2], 1),
    "BabyAI Combined":   ([519.6, 462.8, 8.6], 1),
    "Minihack-5x5":      ([18.8, 18.5, 25.2], 3),
    "Minihack-15x15":    ([14.8, 16.1, 12.3], 3),
    "Minihack-Traps":    ([41.5, 11.6, 17.2], 2),
    "Minihack-Monster":  ([77.2, 19.6, 107.9], 1),
    "Minihack-WoD":      ([144.1, 504.9, 115.5], 0),
}

WC_O4MINI = {  # WorldCoder o4-mini-high (this session)
    "Labyrinth":         (synth(44.2, 2.7), 3),
    "Maze":              (synth(39.6, 5.1), 2),
    "Sokoban":           (synth(63.0, 30.0), 3),
    "BabyAI Pickup":     (synth(72.1, 7.8), 3),
    "BabyAI Unlock":     (synth(99.7, 15.8), 3),
    "BabyAI Combined":   (synth(510.3, 324.0), 1),
    "Minihack-5x5":      (synth(57.4, 5.6), 3),
    "Minihack-15x15":    (synth(155.9, 85.7), 3),
    "Minihack-Traps":    (synth(185.2, 94.1), 3),
    "Minihack-Monster":  (synth(1042.9, 319.9), 1),
    "Minihack-WoD":      (synth(1010.4, 569.4), 0),
}

ORACLE_TIME = 2.13
ORACLE_SUCCESS = 100.0


# ============================================================
# Aggregation
# ============================================================

def aggregate(config):
    """
    Return (mean_time, xerr_seed_std, wins, trials).

    - mean_time: mean of per-task means (of 3 seed walls) → x.
    - xerr:      mean of per-task per-seed sample stdev → typical seed noise.
    - wins/trials: total wins over 33 seed×task combinations.
    """
    task_means = []
    task_seed_stds = []
    total_wins = 0
    for task in TASKS:
        walls, wins = config.get(task, ([0, 0, 0], 0))
        task_means.append(statistics.mean(walls))
        task_seed_stds.append(
            statistics.stdev(walls) if len(walls) > 1 and any(walls) else 0
        )
        total_wins += wins
    return statistics.mean(task_means), statistics.mean(task_seed_stds), total_wins, 33


# (label, data, color, marker, fill_frac)
#
# fill_frac ∈ [0.0, 1.0] represents "how full is the reasoning tank":
#   gpt-4o (no reasoning)      → 0.00  (empty outline)
#   o4-mini-low                → 0.25  (small filled center)
#   o4-mini-medium             → 0.55  (larger filled center)
#   o4-mini-high               → 1.00  (fully filled)
CONFIGS = [
    ('TC (gpt-4o)',                 TC_GPT4O,    'blue',   'D', 0.00),
    ('TC (o4-mini-high)',           TC_O4MINI,   'blue',   'D', 1.00),
    ('TC-P (gpt-4o)',               TCP_GPT4O,   'blue',   'o', 0.00),
    ('TC-P (o4-mini-high)',         TCP_O4MINI,  'blue',   'o', 1.00),
    ('TC-C (gpt-4o)',               TCC_GPT4O,   'blue',   's', 0.00),
    ('TC-C (o4-mini-high)',         TCC_O4MINI,  'blue',   's', 1.00),
    ('LLM+π (gpt-4o)',              LLMPI_GPT4O, 'red',    'D', 0.00),
    ('LLM+π (o4-mini-low)',         LLMPI_LOW,   'red',    'D', 0.25),
    ('LLM+π (o4-mini-medium)',      LLMPI_MED,   'red',    'D', 0.55),
    ('LLM+π (o4-mini-high)',        LLMPI_HIGH,  'red',    'D', 1.00),
    ('LLM+P (gpt-4o)',              LLMP_GPT4O,  'green',  '^', 0.00),
    ('LLM+P (o4-mini-high)',        LLMP_HIGH,   'green',  '^', 1.00),
    ('WorldCoder (gpt-4o)',         WC_GPT4O,    'purple', '^', 0.00),
    ('WorldCoder (o4-mini-high)',   WC_O4MINI,   'purple', '^', 1.00),
]

points = []
for label, d, color, marker, fill_frac in CONFIGS:
    mt, xerr, w, n = aggregate(d)
    p = w / n
    sem = math.sqrt(p * (1 - p) / n) * 100
    points.append({
        'label': label, 'x': mt, 'xerr': xerr, 'y': p * 100, 'yerr': sem,
        'wins': w, 'trials': n, 'color': color, 'marker': marker,
        'fill_frac': fill_frac,
    })

points.append({
    'label': 'Oracle', 'x': ORACLE_TIME, 'xerr': 0,
    'y': ORACLE_SUCCESS, 'yerr': 0, 'wins': 33, 'trials': 33,
    'color': 'orange', 'marker': 'X', 'fill_frac': 1.00,
})

# ============================================================
# Print summary
# ============================================================
print(f"{'Method':<30} {'μ time':>8} {'seed σ':>8}  {'wins':>7}  {'%':>6}")
print('-' * 68)
for p in points:
    print(f"{p['label']:<30} {p['x']:>8.1f} {p['xerr']:>8.1f}  "
          f"{p['wins']:>3}/{p['trials']:<3}  {p['y']:>6.1f}")

# ============================================================
# Plot
# ============================================================
fig, ax = plt.subplots(figsize=(11, 7))

OUTER_SIZE = 13
for p in points:
    ax.errorbar(
        p['x'], p['y'],
        xerr=p['xerr'], yerr=p['yerr'],
        fmt=p['marker'], color=p['color'], ecolor=p['color'],
        elinewidth=1.0, capsize=4, markersize=OUTER_SIZE,
        markerfacecolor='white',
        markeredgewidth=1.4, markeredgecolor=p['color'],
        zorder=3, alpha=0.9,
    )
    if p['fill_frac'] > 0.0:
        inner_size = OUTER_SIZE * math.sqrt(p['fill_frac'])
        ax.plot(
            p['x'], p['y'],
            marker=p['marker'], color=p['color'],
            markersize=inner_size,
            markerfacecolor=p['color'],
            markeredgecolor=p['color'], markeredgewidth=0,
            linestyle='None', zorder=4, alpha=0.95,
        )

ax.set_xlabel('Average Compute Time per Task (seconds)', fontsize=13)
ax.set_ylabel('Success Rate (%)', fontsize=13)
ax.set_xscale('log')
ax.set_xlim(1, 1500)
ax.set_ylim(0, 105)
ax.grid(True, which='both', alpha=0.3)

# One legend entry per configuration.
legend_elems = []
for p in points:
    if p['fill_frac'] == 0.0:
        fs, mfc = 'none', 'white'
    elif p['fill_frac'] < 0.4:
        fs, mfc = 'bottom', p['color']
    elif p['fill_frac'] < 0.75:
        fs, mfc = 'left', p['color']
    else:
        fs, mfc = 'full', p['color']
    legend_elems.append(Line2D(
        [0], [0], marker=p['marker'], color='w',
        markerfacecolor=mfc, markerfacecoloralt='white',
        markeredgecolor=p['color'],
        markersize=11, fillstyle=fs,
        label=p['label'],
    ))
ax.legend(
    handles=legend_elems, loc='center left', bbox_to_anchor=(1.02, 0.5),
    fontsize=8.5, framealpha=0.95, borderaxespad=0,
)

fig.tight_layout()
fig.savefig('compute_time_vs_success_rate.png', dpi=180, bbox_inches='tight')
fig.savefig('compute_time_vs_success_rate.pdf', bbox_inches='tight')
print(f"\nSaved: compute_time_vs_success_rate.png + .pdf")
