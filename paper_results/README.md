# Paper Winners

Canonical artifacts from every winning (and losing) run reported in the
paper. Extracted from the raw sweep dirs via
`../extract_paper_winners.py`.

## Layout

```
paper_winners/
  <method>/                      # full_tc, tc_p, tc_c, llmpi, llmp, worldcoder
    <backbone>/                  # gpt-4o, o4-mini-low, o4-mini-medium, o4-mini-high
      vgdl/
        labyrinth/
          seed_<N>/              # or trial_<N> where paper conventions differ
            worldmodel.py        # LLM-synthesized transition/reward code (WorldCoder)
            predicates.py        # (TC family)
            tc_game/<game>/
              <game>_domain.pddl # PDDL domain
              <game>_0.pddl      # PDDL problem
              <game>_plans.json  # BFS/FD-planned action sequences
            timings_<game>_run_L0.json  # LLM+BFS event log; source of tokens and wall time
            summary.txt          # win/lose status (TC family)
            run_summary.json     # win/lose status (WorldCoder)
        maze/…
        sokoban/…
      babyai/
        seed_<N>/
          level_19/, level_23/, level_13/
          … (same file set)
      minihack/
        5x5/seed_<N>/…
        15x15/seed_<N>/…
        Traps/seed_<N>/…
        Monster/seed_<N>/…
        WoD/seed_<N>/…
  manifest.json                  # every file copied here + source path + size
```

## What's kept per cell

- **Source code** the LLM produced (`worldmodel.py`, `predicates.py`, PDDL)
- **Planner output** (`plans.json`)
- **Instrumentation** (`timings*.json`) — LLM calls, wall time, token usage
- **Outcome** (`summary.txt` / `run_summary.json`)
- **Baseline aggregates** (`aggregate_summary.json` for LLM+π, LLM+P, TC-P)

## What's stripped

The following debug artifacts are NOT copied (regenerable / large):
- `__pycache__/`
- `tape/tape.json` (very large per-step replay)
- `replay_buffers/*.txt` (verbose transitions)
- `steps/*/prompt.txt`, `steps/*/completion_info.json` (raw LLM I/O; occasionally
  useful for post-hoc analysis but ~50 MB per cell)

If you need any of these for a specific cell, they're in the source path
listed in `manifest.json`.

## Convention: seeds

- **VGDL** uses `trial_N` in some paper runs (LLM-noise trials at fixed env
  seed) and `seed_N` in this-session runs. Both are LLM-sampling replicates.
- **BabyAI** uses `seed_N` where `N ∈ {42, 5, 21}` — true env seeds.
- **MiniHack** uses `seed_N` or `trial_N`. Env seed is fixed at 42 for the
  paper convention; the values are LLM-noise replicates. See RESULTS_SUMMARY.md
  for per-batch details.

## Backbone tag meanings

- **`gpt-4o`**: `gpt-4o-2024-11-20`, no reasoning support.
- **`o4-mini-low`**: `o4-mini-2025-04-16` with `reasoning_effort=low`.
- **`o4-mini-medium`**: same model, `reasoning_effort=medium`.
- **`o4-mini-high`**: same model, `reasoning_effort=high` (paper default when o4-mini is used).

## Sanity check

The paper's Main Table, Table A (all-o4-mini-high), and Table B
(all-gpt-4o), as well as the compute-vs-success plot, all derive from
files in this tree. If a plot script fails after `experiments/` is
deleted, first check that its file globs point to `paper_winners/`
rather than the old sweep dirs.
