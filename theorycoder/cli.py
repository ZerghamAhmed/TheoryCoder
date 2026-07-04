"""Command-line interface for TheoryCoder.

This module handles argument parsing and dispatch to the appropriate
execution modes. The main entry point is the `main()` function.
"""
import argparse
import ast
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Plan files per game (edit to your filenames)
PLAN_FILES = {
    "labyrinth":    "labyrinth_plans.json",
    "maze":         "maze_plans.json",
    "sokoban":      "sokoban_plans.json",
    "baba":         "baba_plans.json",
    "doggo":        "doggo_plans.json",
    "drunkdwarf":   "drunkdwarf_plans.json",
    "boulderdash2": "boulderdash2_plans.json",
    "pb1":          "pb1_plans.json",
    "cheesemaze":   "cheesemaze_plans.json",
    "lava":         "lava_plans.json",
    "babyai":       "babyai_plans.json",
    "sokobanFULL":  "sokobanFULL_plans.json",
}


class _Tee(io.TextIOBase):
    """Helper class to tee output to multiple streams."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def start_log(game: str, level_id: int) -> str:
    """Start logging to a file while also printing to console."""
    os.makedirs(os.path.join('run_logs', game), exist_ok=True)
    log_path = os.path.join('run_logs', game, f"{game}_level{level_id}_{int(time.time())}.log")
    _log_file = open(log_path, 'w', buffering=1)  # line-buffered
    sys.stdout = _Tee(sys.__stdout__, _log_file)
    sys.stderr = _Tee(sys.__stderr__, _log_file)
    print(f"[TEE] Logging stdout/stderr to {log_path}")
    return log_path


def clear_run_files(game: str, world_model_name: str, domain_name: str, exp_root: str):
    """Clear generated files for a fresh start."""
    import shutil

    for f in [world_model_name + ".py", domain_name]:
        p = Path(f)
        if p.exists():
            p.unlink()
            print(f"[scratch] removed {p}")

    exp_path = Path(exp_root)
    if exp_path.exists():
        shutil.rmtree(exp_path)
        print(f"[scratch] removed {exp_path}")


def parse_transfer_levels(raw: str) -> Dict[str, set] | set:
    """Parse --transfer-levels argument into set or dict of sets."""
    try:
        tl = ast.literal_eval(raw)
        # normalize to sets
        if isinstance(tl, dict):
            tl = {k: set(v) for k, v in tl.items()}
        elif isinstance(tl, (list, tuple, set)):
            tl = set(tl)
        elif isinstance(tl, int):
            tl = {tl}
        elif not isinstance(tl, set):
            raise ValueError
        return tl
    except Exception:
        raise ValueError(
            "--transfer-levels must be an int, a set/list, or a dict of game -> list/set of ints"
        )


def parse_level_sets(raw: str) -> Dict[str, list]:
    """Parse --level-sets argument into a dictionary."""
    try:
        level_sets = ast.literal_eval(raw)
        assert isinstance(level_sets, dict)
        return level_sets
    except Exception:
        raise ValueError("--level-sets must be a Python dict string, e.g. \"{'labyrinth':[0]}\"")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and return the argument parser for TheoryCoder."""
    parser = argparse.ArgumentParser(
        description="TheoryCoder - Theory-based RL agent for game playing"
    )

    # Game settings
    parser.add_argument('--game', type=str, default='pb1',
                        choices=['baba', 'lava', 'babyai', 'doggo', 'drunkdwarf',
                                 'boulderdash2', 'pb1', 'sokoban', 'labyrinth',
                                 'cheesemaze', 'sokobanFULL'])
    parser.add_argument('--level-sets', type=str, default="{'pb1': [0, 1, 2, 3]}",
                        help="Python dict, e.g. \"{'labyrinth':[0], 'maze':[0,1]}\"")
    parser.add_argument('--episode-length', type=int, default=20)

    # File settings
    parser.add_argument('--world-model-file-name', type=str, default='worldmodel')
    parser.add_argument('--domain-file-name', type=str, default='domain.pddl')
    parser.add_argument('--predicates-file-name', type=str, default='predicates')
    parser.add_argument('--json-reporter-path', type=str,
                        default='KekeCompetition-main/Keke_JS/reports/TBRL_BABA_REPORT.json')

    # Behavior settings
    parser.add_argument('--learn-model', action='store_true')
    parser.add_argument('--query-mode', type=str, default='custom')
    parser.add_argument('--groq-model', type=str, default='llama-3.3-70b-versatile',
                        help='Model name for Groq provider')
    parser.add_argument('--experiment-dir', type=str, default='debuglv3',
                        help='Directory to store experiment runs')
    parser.add_argument('--multi-level', action='store_true',
                        help='Run multiple levels sequentially')
    parser.add_argument('--max-attempts', type=int, default=4,
                        help='Maximum attempts per level')
    parser.add_argument('--prune-plans', action='store_true',
                        help='Use the current method of handling one subplan at a time')
    parser.add_argument('--centralize-files', action='store_true',
                        help='Share domain and predicate files across games')
    parser.add_argument('--scratch', action='store_true',
                        help='Start with a clean slate of generated files')

    # Transfer settings
    parser.add_argument('--transfer-levels', type=str, default='{ "babyai": [13] }',
                        help='Set or per-game dict, e.g. "{13}" or "{ \\"babyai\\":[13] }"')
    parser.add_argument('--transfer-config', type=str, default=None,
                        help='Path to YAML config file for level transfer configuration')

    # Seeding settings
    parser.add_argument('--seed-from', type=str, default=None,
                        help='Path to a previous experiment dir whose artifacts should seed this run')
    parser.add_argument('--seed-src-game', type=str, default=None,
                        help='Name of the SOURCE game folder under tc_game/ to seed from')

    # Trial settings
    parser.add_argument('--num-trials', type=int, default=1,
                        help='Number of times to run the agent (each trial starts fresh)')

    return parser


def build_engine(game: str, level_set: str, level_id: int):
    """Factory function to build game engines."""
    # Import here to avoid circular imports
    from babyai_env import BabyAI
    from doggo_env import DoggoEnv
    from drunkdwarf_env import DrunkDwarfEnv
    from boulderdash2_env import Boulderdash2Env
    from pb1_env import pb1env
    from sokoban_env import SokobanEnv
    from labyrinth_env import LabyrinthEnv
    from cheesemaze_env import CheesemazeEnv
    from sokobanFULL_env import SokobanEnvFULL
    from games import BabaIsYou

    if game == 'baba':
        return BabaIsYou(level_set=level_set, level_id=level_id)
    if game == 'babyai':
        return BabyAI(level_set=level_set, level_id=level_id)
    if game == 'doggo':
        return DoggoEnv(level_set=level_set, level_id=level_id)
    if game == 'drunkdwarf':
        return DrunkDwarfEnv(level_set=level_set, level_id=level_id)
    if game == 'boulderdash2':
        return Boulderdash2Env(level_set=level_set, level_id=level_id)
    if game == 'pb1':
        return pb1env(level_set=level_set, level_id=level_id)
    if game == 'sokoban':
        return SokobanEnv(level_set=level_set, level_id=level_id)
    if game == 'sokobanFULL':
        return SokobanEnvFULL(level_set=level_set, level_id=level_id)
    if game == 'labyrinth':
        return LabyrinthEnv(level_set=level_set, level_id=level_id)
    if game == 'cheesemaze':
        return CheesemazeEnv(level_set=level_set, level_id=level_id)
    raise ValueError(f"Unknown game: {game}")


def run_transfer_config_mode(agent, args, config: dict):
    """Execute transfer config mode with sequential level transfer."""
    from theorycoder.transfer import run_sequential_levels_with_transfer

    # Set debug_no_llm flag on agent
    agent.debug_no_llm = config.get('debug_no_llm', False)
    if agent.debug_no_llm:
        print(f"[DEBUG MODE] Agent will skip all LLM calls")

    # Set fixed_worldmodel path from config if provided
    if 'fixed_worldmodel' in config:
        agent.fixed_worldmodel_path = config['fixed_worldmodel']
        print(f"[FIXED WM] World model fixed to: {agent.fixed_worldmodel_path}")

    # Set planner timeout
    agent.planner_timeout = config.get('planner_timeout', None)
    if agent.planner_timeout:
        print(f"[PLANNER] Timeout set to {agent.planner_timeout} seconds")

    # Initialize game_dir for the agent
    game_name = config['game']
    agent.game_dir = Path(agent.logger.experiment_dir) / "tc_game" / game_name
    agent.game_dir.mkdir(parents=True, exist_ok=True)

    # Set environment variables
    os.environ["TC_WORLDMODEL_FILE"] = str(agent.game_dir / "worldmodel.py")
    os.environ["TC_PREDICATES_FILE"] = str(agent.game_dir / "predicates.py")

    # Handle multi-trial mode
    if args.num_trials > 1:
        print(f"\n{'='*70}")
        print(f"[MULTI-TRIAL MODE] Running {args.num_trials} independent trials")
        print(f"{'='*70}\n")

        trials_dir = Path(agent.logger.experiment_dir) / "transfer_trials"
        trials_dir.mkdir(exist_ok=True)

        all_trial_results = []

        for trial_num in range(1, args.num_trials + 1):
            print(f"\n{'='*70}")
            print(f"[TRIAL {trial_num}/{args.num_trials}]")
            print(f"{'='*70}\n")

            # Create isolated trial directory
            trial_dir = trials_dir / f"trial_{trial_num:03d}"
            trial_dir.mkdir(exist_ok=True)
            (trial_dir / "tape").mkdir(exist_ok=True)
            (trial_dir / "steps").mkdir(exist_ok=True)

            # Setup trial-specific paths
            original_exp_dir = agent.logger.experiment_dir
            agent.logger.experiment_dir = str(trial_dir)
            agent.game_dir = trial_dir / "tc_game" / config['game']
            agent.game_dir.mkdir(parents=True, exist_ok=True)

            os.environ["TC_WORLDMODEL_FILE"] = str(agent.game_dir / "worldmodel.py")
            os.environ["TC_PREDICATES_FILE"] = str(agent.game_dir / "predicates.py")

            # Reset agent state
            agent.tape = [{}]
            agent.logger.tape = []
            agent._init_timing()

            # Run sequential levels for this trial
            trial_results = run_sequential_levels_with_transfer(agent, config, build_engine)
            all_trial_results.append({
                'trial': trial_num,
                'results': trial_results
            })

            # Restore original experiment dir
            agent.logger.experiment_dir = original_exp_dir

        # Summary for multi-trial mode
        print(f"\n{'='*70}")
        print(f"[MULTI-TRIAL SUMMARY]")
        print(f"{'='*70}\n")
        for trial_result in all_trial_results:
            trial_num = trial_result['trial']
            results = trial_result['results']
            successes = sum(1 for r in results if r['success'])
            print(f"Trial {trial_num}: {successes}/{len(results)} levels succeeded")
    else:
        # Single trial mode
        print(f"\n{'='*70}")
        print(f"[SINGLE TRIAL MODE]")
        print(f"{'='*70}\n")

        results = run_sequential_levels_with_transfer(agent, config, build_engine)

        # Summary for single trial
        print(f"\n{'='*70}")
        print(f"[TRANSFER SUMMARY]")
        print(f"{'='*70}\n")
        successes = sum(1 for r in results if r['success'])
        print(f"Overall: {successes}/{len(results)} levels succeeded")
        for r in results:
            status = "✓" if r['success'] else "✗"
            transfer_info = f" (transferred from L{r['transferred_from']})" if r['transferred_from'] else " (learned from scratch)"
            print(f"  {status} Level {r['level']}{transfer_info}")

    # Save tape
    Path('tapes').mkdir(parents=True, exist_ok=True)
    tape_path = f'tapes/transfer_config_{int(time.time())}.json'
    with open(tape_path, 'w') as f:
        json.dump(agent.tape, f, indent=4)
    print(f"\nTape saved to: {tape_path}")


def main():
    """Main entry point for TheoryCoder CLI."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Import here to avoid circular imports at module load time
    # We'll import from theorycoder3 for now, but will update once agent is extracted
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from theorycoder3 import TheoryCoderAgent, seed_game_from_previous
    from theorycoder.transfer import load_transfer_config

    # Parse arguments
    tl = parse_transfer_levels(args.transfer_levels)
    level_sets = parse_level_sets(args.level_sets)

    # Early transfer config loading
    transfer_config = None
    if args.transfer_config:
        print(f"\n{'='*70}")
        print(f"[TRANSFER CONFIG MODE] Loading config from {args.transfer_config}")
        print(f"{'='*70}\n")

        try:
            transfer_config = load_transfer_config(args.transfer_config)
        except Exception as e:
            print(f"[ERROR] Failed to load transfer config: {e}")
            sys.exit(1)

        # Override settings from config
        if 'experiment_dir' in transfer_config:
            args.experiment_dir = transfer_config['experiment_dir']
            print(f"Using experiment dir from config: {args.experiment_dir}")
        if 'game' in transfer_config:
            args.game = transfer_config['game']
            print(f"Using game from config: {args.game}")
        if 'learn_model' in transfer_config:
            args.learn_model = transfer_config['learn_model']
            print(f"Using learn_model from config: {args.learn_model}")
        if 'max_attempts' in transfer_config:
            args.max_attempts = transfer_config['max_attempts']
            print(f"Using max_attempts from config: {args.max_attempts}")
        if 'num_trials' in transfer_config:
            args.num_trials = transfer_config['num_trials']
            print(f"Using num_trials from config: {args.num_trials}")
        if 'query_mode' in transfer_config:
            args.query_mode = transfer_config['query_mode']
            print(f"Using query_mode from config: {args.query_mode}")
        if 'groq_model' in transfer_config:
            args.groq_model = transfer_config['groq_model']
            print(f"Using groq_model from config: {args.groq_model}")
        if 'temperature' in transfer_config:
            args.temperature = transfer_config['temperature']
            print(f"Using temperature from config: {args.temperature}")

        debug_no_llm = transfer_config.get('debug_no_llm', False)
        if debug_no_llm:
            print(f"[DEBUG MODE] No LLM calls - using existing artifacts only")
            args.learn_model = False

    # Scratch cleanup
    exp_root = os.path.join('experiments', args.experiment_dir)
    if args.scratch:
        if args.multi_level:
            for g in level_sets.keys():
                clear_run_files(g, args.world_model_file_name, args.domain_file_name, exp_root)
        else:
            clear_run_files(args.game, args.world_model_file_name, args.domain_file_name, exp_root)

    # Create agent
    agent = TheoryCoderAgent(
        base_dir=exp_root,
        episode_length=args.episode_length,
        world_model_load_name=args.world_model_file_name,
        json_reporter_path=args.json_reporter_path,
        predicates_file_name=args.predicates_file_name,
        domain_file_name=args.domain_file_name,
        do_revise_model=args.learn_model,
        plans_file_name=PLAN_FILES.get(args.game, 'plans.json'),
        query_mode=args.query_mode,
        groq_model=args.groq_model,
        prune_plans=args.prune_plans,
        centralize_files=args.centralize_files,
        create_subdir=False,
        transfer_levels=tl
    )

    # Handle seeding
    if args.seed_from:
        seed_game_from_previous(
            agent,
            args.seed_from,
            args.game,
            copy_plans=True,
            src_game=args.seed_src_game
        )

    # Execute based on mode
    if args.transfer_config:
        run_transfer_config_mode(agent, args, transfer_config)
        sys.exit(0)

    # Multi-level or single-level mode
    if args.multi_level:
        overall = {"levels_completed": [], "levels_failed": []}
        for game, levels in level_sets.items():
            agent.plans_file_name = PLAN_FILES.get(game, 'plans.json')
            for level_id in levels:
                start_log(game, level_id)
                level_set_name = list(level_sets.keys())[0] if len(level_sets) == 1 else (game if game in level_sets else 'default')
                engine = build_engine(game, level_set_name, level_id)
                print(f"\n=== Running {game} level {level_id} ===")
                ok = agent.run(engine, max_attempts=args.max_attempts)
                (overall["levels_completed"] if ok else overall["levels_failed"]).append(f"{game}:{level_id}")

        # Save combined tape
        Path('tapes').mkdir(parents=True, exist_ok=True)
        tape_path = f'tapes/multirun_{int(time.time())}.json'
        with open(tape_path, 'w') as f:
            json.dump(agent.tape, f, indent=4)

        print("\nExperiment Complete!")
        print(f"Levels Completed: {len(overall['levels_completed'])}")
        print(f"Levels Failed: {len(overall['levels_failed'])}")
    else:
        # Single game mode
        level_set_name = list(level_sets.keys())[0]
        level_id = level_sets[level_set_name][0]

        agent.plans_file_name = PLAN_FILES.get(args.game, 'plans.json')

        start_log(args.game, level_id)
        engine = build_engine(args.game, level_set_name, level_id)

        print(f"\n=== Running {args.game} level {level_id} (level_set={level_set_name}) ===")
        agent.run(engine, max_attempts=args.max_attempts)

        # Save tape
        Path('tapes').mkdir(parents=True, exist_ok=True)
        tape_path = f'tapes/{args.game}_{level_set_name}_{level_id}_{int(time.time())}.json'
        with open(tape_path, 'w') as f:
            json.dump(agent.tape, f, indent=4)


if __name__ == '__main__':
    main()
