# Wrappers

One canonical driver per method. All configuration is via environment
variables — no positional args, no per-model duplicates.

## Common env vars

Sourced from `_common.sh`, override on the command line as needed:

| Env var | Default | Meaning |
|---|---|---|
| `MODEL` | `gpt-4o-2024-11-20` | LLM model id |
| `REASONING_EFFORT` | *(empty)* | For o-series only: `low` \| `medium` \| `high` |
| `SEEDS` | `42 5 21` | Space-separated seeds |
| `MAX_REFINEMENTS` | `3` | LLM+π/LLM+P per-attempt retries |
| `MAX_ATTEMPTS` | `3` | TC family per-level attempts |
| `EP_LEN_VGDL` | `100` | VGDL episode length |
| `EP_LEN_BABYAI` | `20` | BabyAI episode length |
| `EP_LEN_MINIHACK` | `10000` | MiniHack episode length |
| `DOMAINS` | `vgdl babyai minihack` | Which of the 3 domains to run |
| `VGDL_GAMES` | `labyrinth maze sokoban` | VGDL games |
| `BABYAI_LEVELS` | `19 23 13` | BabyAI level ids (Pickup, Unlock, Combined) |
| `MH_TASKS` | `5x5:0 15x15:1 WoD:3 Traps:8 Monster:9` | `name:level` pairs |
| `ROOT` | *(auto)* | Output dir tag |

Method-specific extras documented in each script's header.

## Wrappers

| Script | Method | Runner |
|---|---|---|
| `run_full_tc.sh` | Full TC (curriculum + WM + PDDL + revision loop) | `theorycoder2.py`, `theorycoder2_MH.py` |
| `run_tc_c.sh` | TC minus curriculum (every task fresh) | same |
| `run_tc_p.sh` | TC minus revision loop | |
| `run_llmpi.sh` | LLM+π (raw-action sampling) | `baseline_runner.py` |
| `run_llmp.sh` | LLM+P (PDDL synthesis only) | `PDDL_baseline_ALT3_TIME.py`, `..._MH.py` |
| `run_worldcoder.sh` | WorldCoder (Python transition/reward models) | `worldcoder.py` |

## Prereqs

Export before running any wrapper:

```bash
export CUSTOM_BASE_URL="https://go.apis.huit.harvard.edu/ais-openai-direct-limited-schools/v1"
export CUSTOM_API_KEY="…"
```

Conda envs: `tc` (Python 3.10, for VGDL + BabyAI), `minihack` (Python
3.8, for MiniHack — requires `pip install minihack`).

## Examples

```bash
# Full TC on paper baseline (gpt-4o)
./wrappers/run_full_tc.sh

# Full TC on o4-mini at high reasoning effort
MODEL=o4-mini-2025-04-16 REASONING_EFFORT=high ./wrappers/run_full_tc.sh

# LLM+π reasoning sweep (low → medium → high on same model)
for E in low medium high; do
  MODEL=o4-mini-2025-04-16 REASONING_EFFORT=$E ./wrappers/run_llmpi.sh
done

# Only VGDL, only seed 42
DOMAINS=vgdl SEEDS=42 ./wrappers/run_worldcoder.sh

# Force fresh-learn on all MH tasks (skip paper's cherry-picked transfer)
MH_METHODOLOGY=fresh ./wrappers/run_worldcoder.sh
```

## Output convention

Each wrapper writes to a top-level dir tagged by method + backbone:

- Full TC → `tc_<model_tag>/`
- TC-C → `tcc_<model_tag>/`
- LLM+π → `llmpi_<task>_<model_tag>/`
- LLM+P → `llmp_<task>_<model_tag>/`
- WorldCoder → `wc_<model_tag>/`

`<model_tag>` is derived from `MODEL` + `REASONING_EFFORT`
(e.g. `gpt4o`, `o4mini_high`, `o4mini_medium`).

## Skip-guards

All wrappers are resume-safe. On rerun, they detect completed cells and
skip them:

- Log-based methods (Full TC, TC-C) skip if the cell's `.log` file has a
  `SUCCESS`, `WON level`, or `Overall: X/Y levels succeeded` line.
- `run_summary.json`-based methods (WC) skip if the file exists (regardless
  of won/lost — set `ROOT` to a new dir to force a fresh sweep).
- `llm_calls.jsonl`-based methods (LLM+π) skip if the file is non-empty.
- `completed` sentinel files for LLM+P and TC-P (touched after successful
  run).
