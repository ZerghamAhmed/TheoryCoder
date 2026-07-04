#!/usr/bin/env bash
# Full TC (curriculum + world-model + PDDL synthesis + revision loop).
# Runners: theorycoder2.py (VGDL/BabyAI), theorycoder2_MH.py (MH).
#
# Curriculum semantics:
#   VGDL:     labyrinth → maze (transfer from labyrinth) → sokoban (transfer from maze)
#   BabyAI:   L19 → L23 (transfer) → L13 (transfer) — all via a single YAML with
#             `transfer:` blocks (theorycoder2.py --transfer-config).
#   MiniHack: 5x5 (fresh) → {15x15, Traps, Monster} (zero-shot from 5x5) +
#             WoD (fresh, wand mechanic doesn't transfer from 5x5).
#
# Output layout:
#   tc_<model_tag>/
#     vgdl/trial_<S>/{labyrinth,maze,sokoban}/
#     babyai/seed_<S>/level_babyai_{19,23,13}/
#     minihack/seed_<S>/{5x5_lvl0,15x15_lvl1,Traps_lvl8,Monster_lvl9,WoD_lvl3}/

set -uo pipefail
cd "$(dirname "$0")/.."
source wrappers/_common.sh

TAG="$(model_tag)"
ROOT="${ROOT:-tc_${TAG}}"
mkdir -p "$ROOT" "$ROOT/vgdl" "$ROOT/babyai" "$ROOT/minihack"

won_in_log() {
  [ -f "$1" ] && /usr/bin/grep -qE "\[LEVEL [0-9]+\] . SUCCESS|Overall: [0-9]+/[0-9]+ levels succeeded|===== WON level" "$1"
}

# ---------- VGDL — curriculum chain per trial ----------
if should_run_domain vgdl; then
  for TRIAL in $SEEDS; do
    TDIR="$ROOT/vgdl/trial_${TRIAL}"; mkdir -p "$TDIR"
    for GAME in $VGDL_GAMES; do
      SAVE="$TDIR/${GAME}"
      LOG="$SAVE.log"
      won_in_log "$LOG" && { echo "[skip] $SAVE"; continue; }
      echo "== Full TC ${TAG} VGDL $GAME trial=$TRIAL"
      YAML=$(mktemp -t tc_vgdl_${GAME}_XXXX.yaml)
      cat > "$YAML" <<EOF
game: $GAME
experiment_dir: $SAVE
num_trials: 1
learn_model: true
debug_no_llm: false
planner_timeout: 60
max_attempts: $MAX_ATTEMPTS
levels:
  - level: 0
EOF
      # For maze/sokoban seed-from earlier winning game.
      if [ "$GAME" = "maze" ] && [ -d "$TDIR/labyrinth" ]; then
        echo "seed_from: $TDIR/labyrinth" >> "$YAML"
        echo "seed_src_game: labyrinth" >> "$YAML"
      elif [ "$GAME" = "sokoban" ] && [ -d "$TDIR/maze" ]; then
        echo "seed_from: $TDIR/maze" >> "$YAML"
        echo "seed_src_game: maze" >> "$YAML"
      fi
      conda run -n "$ENV_TC" --live-stream python theorycoder2.py \
        --transfer-config "$YAML" \
        --query-mode custom \
        --language-model "$MODEL" $(reasoning_flag) \
        --episode-length "$EP_LEN_VGDL" \
        --seed "$TRIAL" 2>&1 | /usr/bin/tee "$LOG"
    done
  done
fi

# ---------- BabyAI — L19 → L23 → L13 chain via in-YAML transfer ----------
if should_run_domain babyai; then
  for SEED in $SEEDS; do
    BDIR="$ROOT/babyai/seed_${SEED}"
    LOG="$BDIR.log"
    won_in_log "$LOG" && { echo "[skip] $BDIR"; continue; }
    echo "== Full TC ${TAG} BabyAI seed=$SEED (L19→L23→L13)"
    YAML=$(mktemp -t tc_babyai_${SEED}_XXXX.yaml)
    cat > "$YAML" <<EOF
game: babyai
experiment_dir: $BDIR
num_trials: 1
learn_model: true
debug_no_llm: false
planner_timeout: 45
max_attempts: $MAX_ATTEMPTS
levels:
  - level: 19
  - level: 23
    transfer:
      from_level: 19
      worldmodel: true
      domain: "extend"
      predicates: "extend"
      problem: false
  - level: 13
    transfer:
      from_level: 23
      worldmodel: true
      domain: true
      predicates: true
      problem: false
EOF
    conda run -n "$ENV_TC" --live-stream python theorycoder2.py \
      --transfer-config "$YAML" \
      --query-mode custom \
      --language-model "$MODEL" $(reasoning_flag) \
      --episode-length "$EP_LEN_BABYAI" \
      --seed "$SEED" 2>&1 | /usr/bin/tee "$LOG"
  done
fi

# ---------- MiniHack — 5x5 fresh → 0-shot transfers + WoD fresh ----------
if should_run_domain minihack; then
  run_mh () {
    local lvl=$1; local exp_dir=$2; local learn_flag=$3; local seed_from=$4
    local extra=""
    [ -n "$seed_from" ] && extra="--seed-from $seed_from --seed-src-game minihack"
    conda run -n "$ENV_MH" --live-stream python theorycoder2_MH.py \
      --game minihack --level-sets "{'minihack': [$lvl]}" \
      --query-mode custom \
      --language-model "$MODEL" $(reasoning_flag) \
      --max-attempts "$MAX_ATTEMPTS" \
      --episode-length "$EP_LEN_MINIHACK" \
      $learn_flag \
      --experiment-dir "$exp_dir" \
      $extra 2>&1 | /usr/bin/tee "${exp_dir}.log"
  }
  for SEED in $SEEDS; do
    mkdir -p "$ROOT/minihack/seed_${SEED}"   # ensure per-task .log dir exists for tee/won_in_log
    # Step 1: 5x5 fresh
    FIVE="$ROOT/minihack/seed_${SEED}/5x5_lvl0"
    if ! won_in_log "${FIVE}.log"; then
      echo "== Full TC ${TAG} MiniHack 5x5 seed=$SEED (fresh)"
      run_mh 0 "$FIVE" "--learn-model" ""
    fi
    # Steps 2-4: 0-shot transfers if 5x5 won
    if won_in_log "${FIVE}.log"; then
      for SPEC in "1:15x15" "8:Traps" "9:Monster"; do
        LVL="${SPEC%%:*}"; NAME="${SPEC##*:}"
        OUT="$ROOT/minihack/seed_${SEED}/${NAME}_lvl${LVL}"
        won_in_log "${OUT}.log" && { echo "[skip] $OUT"; continue; }
        echo "== Full TC ${TAG} MiniHack $NAME seed=$SEED (0-shot from 5x5)"
        run_mh "$LVL" "$OUT" "" "experiments/${FIVE}"
      done
    else
      echo "[warn] 5x5 seed=$SEED did not win → skipping 0-shot transfers"
    fi
    # Step 5: WoD fresh
    WOD="$ROOT/minihack/seed_${SEED}/WoD_lvl3"
    if ! won_in_log "${WOD}.log"; then
      echo "== Full TC ${TAG} MiniHack WoD seed=$SEED (fresh)"
      run_mh 3 "$WOD" "--learn-model" ""
    fi
  done
fi

echo
echo "Full TC ${TAG} sweep complete → $ROOT"
