#!/usr/bin/env bash
# TC-C (TheoryCoder minus curriculum): every task learned fresh, no transfer.
# Runners: theorycoder2.py (VGDL/BabyAI), theorycoder2_MH.py (MH).

set -uo pipefail
cd "$(dirname "$0")/.."
source wrappers/_common.sh

TAG="$(model_tag)"
ROOT="${ROOT:-tcc_${TAG}}"
mkdir -p "$ROOT" "$ROOT/vgdl" "$ROOT/babyai" "$ROOT/minihack"

won_in_log() {
  [ -f "$1" ] && /usr/bin/grep -qE "\[LEVEL [0-9]+\] . SUCCESS|Overall: [0-9]+/[0-9]+ levels succeeded|===== WON level" "$1"
}

# ---------- VGDL — each game fresh ----------
if should_run_domain vgdl; then
  for GAME in $VGDL_GAMES; do
    for TRIAL in $SEEDS; do
      SAVE="$ROOT/vgdl/${GAME}/trial_${TRIAL}"
      LOG="$SAVE.log"
      won_in_log "$LOG" && { echo "[skip] $SAVE"; continue; }
      echo "== TC-C ${TAG} VGDL $GAME trial=$TRIAL"
      YAML=$(mktemp -t tcc_vgdl_XXXX.yaml)
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
      conda run -n "$ENV_TC" --live-stream python theorycoder2.py \
        --transfer-config "$YAML" \
        --query-mode custom \
        --language-model "$MODEL" $(reasoning_flag) \
        --episode-length "$EP_LEN_VGDL" \
        --seed "$TRIAL" 2>&1 | /usr/bin/tee "$LOG"
    done
  done
fi

# ---------- BabyAI — each level fresh ----------
if should_run_domain babyai; then
  for SEED in $SEEDS; do
    for LEVEL in $BABYAI_LEVELS; do
      SAVE="$ROOT/babyai/seed_${SEED}/level_${LEVEL}"
      LOG="$SAVE.log"
      won_in_log "$LOG" && { echo "[skip] $SAVE"; continue; }
      echo "== TC-C ${TAG} BabyAI seed=$SEED L$LEVEL"
      YAML=$(mktemp -t tcc_babyai_XXXX.yaml)
      cat > "$YAML" <<EOF
game: babyai
experiment_dir: $SAVE
num_trials: 1
learn_model: true
debug_no_llm: false
planner_timeout: 45
max_attempts: $MAX_ATTEMPTS
levels:
  - level: $LEVEL
EOF
      conda run -n "$ENV_TC" --live-stream python theorycoder2.py \
        --transfer-config "$YAML" \
        --query-mode custom \
        --language-model "$MODEL" $(reasoning_flag) \
        --episode-length "$EP_LEN_BABYAI" \
        --seed "$SEED" 2>&1 | /usr/bin/tee "$LOG"
    done
  done
fi

# ---------- MiniHack — each task fresh (--learn-model, no --seed-from) ----------
if should_run_domain minihack; then
  for SPEC in $MH_TASKS; do
    NAME="${SPEC%%:*}"; LVL="${SPEC##*:}"
    for SEED in $SEEDS; do
      SAVE="$ROOT/minihack/${NAME}_lvl${LVL}/trial_${SEED}"
      LOG="$SAVE.log"
      won_in_log "$LOG" && { echo "[skip] $SAVE"; continue; }
      echo "== TC-C ${TAG} MiniHack $NAME (lvl $LVL) trial=$SEED"
      conda run -n "$ENV_MH" --live-stream python theorycoder2_MH.py \
        --game minihack --level-sets "{'minihack': [$LVL]}" \
        --query-mode custom \
        --language-model "$MODEL" $(reasoning_flag) \
        --max-attempts "$MAX_ATTEMPTS" \
        --episode-length "$EP_LEN_MINIHACK" \
        --learn-model \
        --experiment-dir "$SAVE" 2>&1 | /usr/bin/tee "$LOG"
    done
  done
fi

echo
echo "TC-C ${TAG} sweep complete → $ROOT"
