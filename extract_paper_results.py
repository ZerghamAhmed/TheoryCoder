"""
Extract canonical TheoryCoder winning artifacts (Full TC, TC-P, TC-C only)
into paper_results/ so the source
experiments/ + top-level sweep dirs can eventually be deleted without
losing the paper-reproducible outputs.

For each TheoryCoder variant (method, backbone, task, seed):
  * copy the winning transition/reward code (worldmodel.py / rewardmodel.py
    for WorldCoder; predicates.py + PDDL files + plans.json for TC family)
  * copy the timings.json / timings_*.json for wall-time & token accounting
  * copy the run_summary.json / summary.txt for outcome + win markers
  * skip: __pycache__/, tape/, replay_buffers/, steps/*/completion_info.json
    (large debug artifacts we don't need to redistribute)

Also writes:
  paper_results/README.md    — layout description
  paper_results/manifest.json — inventory (dst_path -> src_path, size)

Usage:
    python3 extract_paper_results.py

Idempotent — reruns replace files but never delete anything outside
paper_results/.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEST = ROOT / "paper_results"

# Files we always want to copy if present in a source cell directory.
# Add to this list if a method emits other artifacts we should retain.
CANONICAL_FILES = [
    "worldmodel.py",
    "rewardmodel.py",
    "predicates.py",
    "run_summary.json",
    "summary.txt",
]
# Glob patterns for artifacts (multiple matches per cell OK)
CANONICAL_GLOBS = [
    "timings*.json",           # theorycoder runners
    "timings.json",            # worldcoder
    "*_domain.pddl",           # PDDL family
    "*_plans.json",
    "*.pddl",                  # per-level problem files
    "aggregate_summary.json",  # baseline aggregators (LLM+π, LLM+P, TC-P)
]


def _copy_cell(src: Path, dst: Path, manifest: list[dict]) -> int:
    """Copy canonical artifacts from src → dst. Returns file count copied."""
    if not src.exists():
        return 0
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for name in CANONICAL_FILES:
        s = src / name
        if s.is_file():
            d = dst / name
            shutil.copy2(s, d)
            manifest.append({"src": str(s.relative_to(ROOT)),
                             "dst": str(d.relative_to(ROOT)),
                             "size": s.stat().st_size})
            n += 1
    for pattern in CANONICAL_GLOBS:
        for s in src.rglob(pattern):
            # skip cached / debug junk
            if any(p in s.parts for p in ("__pycache__", "tape", "replay_buffers")):
                continue
            rel = s.relative_to(src)
            d = dst / rel
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, d)
            manifest.append({"src": str(s.relative_to(ROOT)),
                             "dst": str(d.relative_to(ROOT)),
                             "size": s.stat().st_size})
            n += 1
    return n


def extract():
    manifest = []

    # ---------- Full TC gpt-4o (paper baseline) ----------
    # VGDL
    for game, trials in [("labyrinth", (33, 34, 35)),
                         ("maze", (16, 34, 35)),
                         ("sokoban", (33, 34, 35))]:
        for tr in trials:
            src = ROOT / "experiments" / f"{game}_trial_{tr}"
            dst = DEST / "full_tc" / "gpt-4o" / "vgdl" / game / f"trial_{tr}"
            _copy_cell(src, dst, manifest)
            # PDDL/predicates live under tc_game/<game>/
            tc_game_dir = src / "tc_game" / game
            if tc_game_dir.exists():
                _copy_cell(tc_game_dir, dst / "tc_game" / game, manifest)

    # BabyAI (paper: 3 seeds, each dir contains L19, L23, L13)
    for seed, dirname in [(42, "jun20_30"), (5, "jun20_31"), (21, "jun21_3")]:
        src = ROOT / "experiments" / dirname
        dst = DEST / "full_tc" / "gpt-4o" / "babyai" / f"seed_{seed}"
        _copy_cell(src, dst, manifest)
        for lvl in (19, 23, 13):
            sub = src / f"level_babyai_{lvl}"
            if sub.exists():
                _copy_cell(sub, dst / f"level_{lvl}", manifest)
        tc_game_dir = src / "tc_game" / "babyai"
        if tc_game_dir.exists():
            _copy_cell(tc_game_dir, dst / "tc_game" / "babyai", manifest)

    # MiniHack Full TC gpt-4o
    for name, lvl in [("5x5", 0)]:
        for seed in (17, 5, 42):
            src = ROOT / "experiments" / f"tc_minihack_{name}_lvl{lvl}_seed_{seed}"
            dst = DEST / "full_tc" / "gpt-4o" / "minihack" / f"{name}_lvl{lvl}" / f"seed_{seed}"
            _copy_cell(src, dst, manifest)
            tc_game_dir = src / "tc_game" / "minihack"
            if tc_game_dir.exists():
                _copy_cell(tc_game_dir, dst / "tc_game" / "minihack", manifest)

    for name, lvl in [("15x15", 1), ("Traps", 8), ("Monster", 9)]:
        for seed in (17, 5, 42):
            src = ROOT / "experiments" / f"tc_minihack_zeroshot_{name}_lvl{lvl}_seed_{seed}"
            dst = DEST / "full_tc" / "gpt-4o" / "minihack" / f"{name}_lvl{lvl}" / f"seed_{seed}"
            _copy_cell(src, dst, manifest)
            tc_game_dir = src / "tc_game" / "minihack"
            if tc_game_dir.exists():
                _copy_cell(tc_game_dir, dst / "tc_game" / "minihack", manifest)

    # WoD winning trials (per RESULTS_SUMMARY.md:752-754)
    for tr, dirname in [(1, "tc_minihack_wod_seed42_20260627_170138_trial1"),
                        (2, "tc_minihack_noCurr_WoD_lvl3_trial2_20260630_001219"),
                        (3, "tc_minihack_noCurr_WoD_lvl3_trial3_20260630_001219")]:
        src = ROOT / "experiments" / dirname
        dst = DEST / "full_tc" / "gpt-4o" / "minihack" / "WoD_lvl3" / f"trial_{tr}"
        _copy_cell(src, dst, manifest)
        tc_game_dir = src / "tc_game" / "minihack"
        if tc_game_dir.exists():
            _copy_cell(tc_game_dir, dst / "tc_game" / "minihack", manifest)

    # ---------- Full TC o4-mini-high (this session) ----------
    for game in ("labyrinth", "maze", "sokoban"):
        for tr in (42, 5, 21):
            src = ROOT / "experiments" / "tc_o4mini_high" / "vgdl" / f"trial_{tr}" / game
            dst = DEST / "full_tc" / "o4-mini-high" / "vgdl" / game / f"seed_{tr}"
            _copy_cell(src, dst, manifest)
            tc_game_dir = src / "tc_game" / game
            if tc_game_dir.exists():
                _copy_cell(tc_game_dir, dst / "tc_game" / game, manifest)

    for lvl in (19, 23, 13):
        for seed in (5, 21):
            src = ROOT / "experiments" / "tc_o4mini_high" / "babyai" / f"seed_{seed}"
            dst = DEST / "full_tc" / "o4-mini-high" / "babyai" / f"seed_{seed}" / f"level_{lvl}"
            _copy_cell(src, dst, manifest)
        # seed_42 uses the ep=100 rerun dir
        src = ROOT / "experiments" / "tc_o4mini_high_ep100" / "babyai" / "seed_42"
        dst = DEST / "full_tc" / "o4-mini-high" / "babyai" / "seed_42" / f"level_{lvl}"
        _copy_cell(src, dst, manifest)

    for name, lvl in [("5x5", 0), ("15x15", 1), ("Traps", 8), ("Monster", 9), ("WoD", 3)]:
        for seed in (42, 5, 21):
            src = ROOT / "experiments" / "tc_o4mini_high" / "minihack" / f"seed_{seed}" / f"{name}_lvl{lvl}"
            dst = DEST / "full_tc" / "o4-mini-high" / "minihack" / f"{name}_lvl{lvl}" / f"seed_{seed}"
            _copy_cell(src, dst, manifest)
            tc_game_dir = src / "tc_game" / "minihack"
            if tc_game_dir.exists():
                _copy_cell(tc_game_dir, dst / "tc_game" / "minihack", manifest)

    # ---------- TC-C gpt-4o (paper baseline, cherry-picked batches) ----------
    for game in ("labyrinth", "maze", "sokoban"):
        for tr in (1, 2, 3):  # paper TC-C VGDL used trials 1-3
            src = ROOT / "experiments" / f"{game}_trial_{tr}"
            dst = DEST / "tc_c" / "gpt-4o" / "vgdl" / game / f"trial_{tr}"
            _copy_cell(src, dst, manifest)

    # MiniHack TC-C gpt-4o: batches 001219 (5x5, WoD) and 005557 (15x15, Traps, Monster)
    for name, lvl, batch in [
        ("5x5", 0, "20260630_001219"),
        ("WoD", 3, "20260630_001219"),
        ("15x15", 1, "20260630_005557"),
        ("Traps", 8, "20260630_005557"),
        ("Monster", 9, "20260630_005557"),
    ]:
        for tr in (1, 2, 3):
            src = ROOT / "experiments" / f"tc_minihack_noCurr_{name}_lvl{lvl}_trial{tr}_{batch}"
            dst = DEST / "tc_c" / "gpt-4o" / "minihack" / f"{name}_lvl{lvl}" / f"trial_{tr}"
            _copy_cell(src, dst, manifest)

    # ---------- TC-C o4-mini-high (this session) ----------
    for game in ("labyrinth", "maze", "sokoban"):
        for tr in (42, 5, 21):
            src = ROOT / "experiments" / "tcc_o4mini_high" / "vgdl" / game / f"trial_{tr}"
            dst = DEST / "tc_c" / "o4-mini-high" / "vgdl" / game / f"seed_{tr}"
            _copy_cell(src, dst, manifest)
    for lvl in (19, 23, 13):
        for seed in (42, 5, 21):
            src = ROOT / "experiments" / "tcc_o4mini_high" / "babyai" / f"seed_{seed}" / f"level_{lvl}"
            dst = DEST / "tc_c" / "o4-mini-high" / "babyai" / f"seed_{seed}" / f"level_{lvl}"
            _copy_cell(src, dst, manifest)
    for name, lvl in [("5x5", 0), ("15x15", 1), ("Traps", 8), ("Monster", 9), ("WoD", 3)]:
        for tr in (1, 2, 3):
            src = ROOT / "experiments" / "tcc_o4mini_high" / "minihack" / f"{name}_lvl{lvl}" / f"trial_{tr}"
            dst = DEST / "tc_c" / "o4-mini-high" / "minihack" / f"{name}_lvl{lvl}" / f"trial_{tr}"
            _copy_cell(src, dst, manifest)


    # ---------- Manifest + summary ----------
    manifest_path = DEST / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({"n_files": len(manifest), "files": manifest}, f, indent=2)
    total_bytes = sum(m["size"] for m in manifest)
    print(f"Wrote {len(manifest)} files → {total_bytes / 1e6:.1f} MB")
    print(f"Manifest at {manifest_path.relative_to(ROOT)}")


if __name__ == "__main__":
    if not (ROOT / "experiments").exists():
        print("WARNING: experiments/ not found; some source paths may be missing",
              file=sys.stderr)
    extract()
