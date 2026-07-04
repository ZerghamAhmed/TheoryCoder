# TheoryCoder-2

Code + canonical artifacts for **"TheoryCoder-2"**: learning transferable
world models by jointly synthesising PDDL abstractions and Python
transition functions with an LLM.

---

## Repository layout

```
theorycoder_local/
  wrappers/                 # unified sweep drivers
    _common.sh              # env var config shared by all wrappers
    run_full_tc.sh          # Full TC
    run_tc_c.sh             # TC-C (no curriculum)
    README.md
  paper_results/            # canonical winning artifacts per (backbone, task, seed)
    README.md
    manifest.json           # traceable dst → src map
    full_tc/gpt-4o/…
    full_tc/o4-mini-high/…
    tc_c/…
  abstraction_prompts/      # per-game PDDL synthesis prompts
  all_games/                # VGDL game specs + level files (VGDLEnvAndres reads here)
  KekeCompetition-main/     # Baba is You game engine (used by games.py)
  envs/                     # game envs (BabyAI, MiniHack, VGDL wrappers, state converters)

  theorycoder2.py           # runner (VGDL, BabyAI)
  theorycoder2_MH.py        # runner (MiniHack)

  planner.py, preprocessing.py, levelrunner.py, utils.py, experiment_logger.py

  plot_compute_time.py      # regenerates the compute-vs-success plot (standalone)
  extract_paper_results.py  # rebuilds paper_results/ from raw sweep dirs
  compute_time_vs_success_rate.{png,pdf}   # paper figure

  RESULTS_SUMMARY.md        # canonical results with provenance
  paper_notes.md            # BabyAI experiment notes

  requirements_theorycoder.txt       # conda env "theorycoder" (VGDL + BabyAI, Python 3.10)
  requirements_minihack.txt # conda env "minihack" (MiniHack, Python 3.8)
```

---

## Set up

You need **two conda envs** — one for VGDL + BabyAI (Python 3.10), one for
MiniHack (Python 3.8, MiniHack pins to that).

```bash
# VGDL + BabyAI
conda create -n theorycoder python=3.10 -y
conda activate theorycoder
pip install -r requirements_theorycoder.txt

# MiniHack
conda create -n minihack python=3.8 -y
conda activate minihack
pip install -r requirements_minihack.txt
```

The wrappers dispatch to the right env automatically — no manual
activation needed once created.

### Fast-Downward

TheoryCoder calls the Fast-Downward planner. Point to your local build
via `downward_config.yaml`:

```yaml
fast_downward_path: /path/to/fast-downward/fast-downward.py
```

### LLM gateway

All wrappers assume an OpenAI-compatible endpoint. Export before running:

```bash
export CUSTOM_BASE_URL="https://…"       # your endpoint
export CUSTOM_API_KEY="…"
```

---

## Running experiments

The two headline runs demonstrate what TheoryCoder-2 does:

### VGDL curriculum sweep (Labyrinth → Maze → Sokoban)

Learn Labyrinth from scratch, then transfer the learned world model to
Maze, then transfer again to Sokoban. Runs 3 seeds by default.

```bash
# gpt-4o (paper Main table default)
./wrappers/run_full_tc.sh

# Or on a reasoning model:
MODEL=o4-mini-2025-04-16 REASONING_EFFORT=high ./wrappers/run_full_tc.sh

# Just VGDL, single seed (smoke test):
DOMAINS=vgdl SEEDS=42 ./wrappers/run_full_tc.sh
```

Output goes to `tc_<model_tag>/vgdl/trial_<seed>/{labyrinth,maze,sokoban}/`.
Each cell contains the learned PDDL domain, problem file, Python world
model, plans, timings, and a summary of win/loss.

### BabyAI transfer sequence (Pickup → Unlock → Combined Skills)

Chains L19 (Pickup) → L23 (Unlock) → L13 (Combined Skills) with
in-YAML `transfer:` blocks so downstream levels inherit the learned
world model from earlier ones.

```bash
DOMAINS=babyai ./wrappers/run_full_tc.sh

# Single seed smoke test:
DOMAINS=babyai SEEDS=42 ./wrappers/run_full_tc.sh
```

Output: `tc_<model_tag>/babyai/seed_<seed>/level_babyai_{19,23,13}/`.

### MiniHack (5x5 fresh → zero-shot transfers + WoD)

Learn the 5x5 room from scratch, then zero-shot transfer the learned
world model to 15x15, Traps, and Monster (which share dynamics).
WoD (Wand of Death) requires learning new mechanics from scratch.
Uses the second conda env (`minihack`).

```bash
DOMAINS=minihack ./wrappers/run_full_tc.sh

# Single seed smoke test:
DOMAINS=minihack SEEDS=42 ./wrappers/run_full_tc.sh
```

Output:
`tc_<model_tag>/minihack/seed_<seed>/{5x5_lvl0,15x15_lvl1,Traps_lvl8,Monster_lvl9,WoD_lvl3}/`

Runner: `theorycoder2_MH.py` (separate from `theorycoder2.py` for VGDL/BabyAI).

### Ablation: no curriculum (TC-C)

Same tasks but every level is learned fresh — no transfer between
levels:

```bash
./wrappers/run_tc_c.sh                                 # all 33 cells
DOMAINS=vgdl SEEDS=42 ./wrappers/run_tc_c.sh          # single-seed smoke
```

### Configuration

All wrappers are env-var configurable. See `wrappers/README.md` for the
full reference, but the common overrides are:

| Env var | Default | Meaning |
|---|---|---|
| `MODEL` | `gpt-4o-2024-11-20` | LLM model id |
| `REASONING_EFFORT` | *(empty)* | For o-series only: `low` / `medium` / `high` |
| `SEEDS` | `42 5 21` | Space-separated seeds |
| `DOMAINS` | `vgdl babyai minihack` | Which domains to run |
| `EP_LEN_VGDL` | `100` | VGDL episode length |
| `EP_LEN_BABYAI` | `20` | BabyAI episode length |
| `EP_LEN_MINIHACK` | `10000` | MiniHack episode length |

All wrappers are **resume-safe**: they skip cells whose logs show a win
marker, so you can Ctrl-C and re-invoke.

---

## Reproducing paper figures

### Compute-vs-success plot

Standalone — no external data dependency:

```bash
python3 plot_compute_time.py
# → compute_time_vs_success_rate.{png,pdf}
```

Numbers are hardcoded in `plot_compute_time.py` and match the paper's
per-configuration values. Edit the `CONFIGS` dict to add a new one.

### Rebuild `paper_results/` from a fresh sweep

```bash
python3 extract_paper_results.py
```

Walks the top-level sweep dirs produced by the wrappers and copies the
canonical artifacts + writes `manifest.json`.

---

## Layout notes for extending the code

- **Add a new LLM backbone**: modify the `is_reasoning` detection block
  in `theorycoder2.py` (around line 747). The wrapper's `model_tag()`
  helper in `wrappers/_common.sh` also needs a case for the new model.
- **Add a new method**: create a wrapper in `wrappers/` sourcing
  `_common.sh`, following the pattern of the existing ones.
- **Add a new domain**: add a runner branch (VGDL/BabyAI/MiniHack) to
  each wrapper and add the domain to `DOMAINS` in `_common.sh`.

---

## Acknowledgements

This repo builds off of https://github.com/c-j-bates/model-based-rl-with-llms.
