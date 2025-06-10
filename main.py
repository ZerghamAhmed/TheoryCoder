#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

from theorycoder import TheoryCoderAgent
from envs.babyai_env import BabyAI
from envs.pb1_env import pb1env
from envs.sokoban_env import SokobanEnv
from envs.labyrinth_env import LabyrinthEnv
from games import BabaIsYou
from envs.maze_env import MazeEnv
from envs.cheesemaze_env import CheesemazeEnv


def init_engine(game, level_set, level_id):
    if game == 'baba':
        return BabaIsYou(level_set=level_set, level_id=level_id)
    elif game == 'lava':
        from games import LavaGrid
        return LavaGrid()
    elif game == 'babyai':
        return BabyAI(level_set=level_set, level_id=level_id)
    elif game == 'pb1':
        return pb1env(level_set=level_set, level_id=level_id)
    elif game == 'sokoban':
        return SokobanEnv(level_set=level_set, level_id=level_id)
    elif game == 'labyrinth':
        return LabyrinthEnv(level_set=level_set, level_id=level_id)
    elif game == 'cheesemaze':
        return CheesemazeEnv(level_set=level_set, level_id=level_id)
    else:
        raise ValueError(f"Unsupported game: {game}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, default='pb1')
    parser.add_argument('--level-sets', type=str, default="{'pb1': [0, 1, 2, 3]}")
    parser.add_argument('--episode-length', type=int, default=20)
    parser.add_argument('--world-model-file-name', type=str, default='worldmodel.py')
    parser.add_argument('--domain-file-name', type=str, default='domain.pddl')
    parser.add_argument('--predicates-file-name', type=str, default='predicates.py')
    parser.add_argument('--learn-model', action='store_true')
    parser.add_argument('--query-mode', type=str, default='groq')
    parser.add_argument('--experiment-dir', type=str, default='debuglv3')
    parser.add_argument('--multi-level', action='store_true')
    parser.add_argument('--max-attempts', type=int, default=4)
    parser.add_argument('--prune-plans', action='store_true')
    args = parser.parse_args()

    # parse level-sets string into dict
    level_sets = eval(args.level_sets)

    # derive a default plans file based on game
    plan_file_name = f"{args.game}_plans.json"

    # instantiate agent
    agent = TheoryCoderAgent(
        world_model_load_name=args.world_model_file_name,
        language_model='gpt-4o',
        domain_file_name=args.domain_file_name,
        predicates_file_name=args.predicates_file_name,
        query_mode=args.query_mode,
        do_revise_model=args.learn_model,
        plans_file_name=plan_file_name,
        base_dir=args.experiment_dir,
        experiment_name=None,
        prune_plans=args.prune_plans
    )

    if args.multi_level:
        results = agent.run_multiple_levels(level_sets, max_revisions=5, max_attempts=args.max_attempts)
        print("\nExperiment Complete!")
        print(f"Levels Completed: {len(results['levels_completed'])}")
        print(f"Levels Failed: {len(results['levels_failed'])}")
    else:
        # single-level mode: pick first
        game = args.game
        level_set = list(level_sets.keys())[0]
        level_id = level_sets[level_set][0]
        engine = init_engine(game, level_set, level_id)
        agent.run(engine, max_revisions=5, max_attempts=args.max_attempts)

        # save the tape
        tape_dir = Path(args.experiment_dir) / 'tapes'
        tape_dir.mkdir(parents=True, exist_ok=True)
        tape_path = tape_dir / f"{args.game}_{level_set}_{level_id}_{int(time.time())}.json"
        with open(tape_path, 'w') as f:
            json.dump(agent.tape, f, indent=4)


if __name__ == '__main__':
    main()
