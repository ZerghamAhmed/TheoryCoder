# Shared config for all method wrappers.
# Sourced by every run_*.sh in this dir.
#
# Environment variables (override on the command line):
#   MODEL             = LLM to hit (default: gpt-4o-2024-11-20)
#   REASONING_EFFORT  = for o-series only (low/medium/high). Empty = no reasoning.
#   SEEDS             = space-separated seed list (default: "42 5 21")
#   MAX_REFINEMENTS   = LLM+π/LLM+P per-attempt retries (default: 3)
#   MAX_ATTEMPTS      = TC family per-level attempts (default: 3)
#   EP_LEN_VGDL       = episode length for VGDL (default: 100)
#   EP_LEN_BABYAI     = episode length for BabyAI (default: 20)
#   EP_LEN_MINIHACK   = episode length for MiniHack (default: 10000)
#   DOMAINS           = which domains to run (default: "vgdl babyai minihack")
#   VGDL_GAMES        = default: "labyrinth maze sokoban"
#   BABYAI_LEVELS     = default: "19 23 13" (Pickup Unlock Combined)
#   MH_TASKS          = default: "5x5:0 15x15:1 WoD:3 Traps:8 Monster:9"
#                       (colon-separated name:level pairs)
#   ROOT              = output dir tag (default: derived from METHOD + MODEL)
#
# Method-specific env vars (documented in each run_*.sh):
#   METHOD, RUNNER (Python entry point), etc.

: "${CUSTOM_BASE_URL:?CUSTOM_BASE_URL must be exported}"
: "${CUSTOM_API_KEY:?CUSTOM_API_KEY must be exported}"

MODEL="${MODEL:-gpt-4o-2024-11-20}"
REASONING_EFFORT="${REASONING_EFFORT:-}"
SEEDS="${SEEDS:-42 5 21}"
MAX_REFINEMENTS="${MAX_REFINEMENTS:-3}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
EP_LEN_VGDL="${EP_LEN_VGDL:-100}"
EP_LEN_BABYAI="${EP_LEN_BABYAI:-20}"
EP_LEN_MINIHACK="${EP_LEN_MINIHACK:-10000}"
DOMAINS="${DOMAINS:-vgdl babyai minihack}"
VGDL_GAMES="${VGDL_GAMES:-labyrinth maze sokoban}"
BABYAI_LEVELS="${BABYAI_LEVELS:-19 23 13}"
MH_TASKS="${MH_TASKS:-5x5:0 15x15:1 WoD:3 Traps:8 Monster:9}"

# Conda envs
ENV_TC="theorycoder"
ENV_MH="minihack"

# Derive a short model tag for output paths.
model_tag() {
  case "$MODEL" in
    gpt-4o*)          echo "gpt4o" ;;
    o4-mini*)         echo "o4mini${REASONING_EFFORT:+_$REASONING_EFFORT}" ;;
    o1*)              echo "o1${REASONING_EFFORT:+_$REASONING_EFFORT}" ;;
    *)                echo "$MODEL" ;;
  esac
}

# Common Python runner flags for methods that support reasoning_effort.
reasoning_flag() {
  if [ -n "$REASONING_EFFORT" ]; then
    echo "--reasoning-effort $REASONING_EFFORT"
  fi
}

# Domain gate: skip a domain if not in DOMAINS list.
should_run_domain() {
  local d="$1"
  case " $DOMAINS " in
    *" $d "*) return 0 ;;
    *)        return 1 ;;
  esac
}
