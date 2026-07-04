"""Transfer logic for level-to-level artifact sharing.

This module handles loading transfer configurations from YAML files
and executing sequential levels with configurable artifact transfer.
"""
import shutil
import sys
from pathlib import Path
from typing import Optional, Callable, Dict, List, Any
import yaml


def load_transfer_config(config_path: str) -> dict:
    """Load and validate YAML transfer configuration.

    Args:
        config_path: Path to YAML config file

    Returns:
        Parsed config dict with validated structure

    Raises:
        ValueError: If config is invalid
        FileNotFoundError: If file doesn't exist
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Transfer config file not found: {config_path}")

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except Exception as e:
        raise ValueError(f"Failed to parse YAML config: {e}")

    # Validate required fields
    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML dict")

    required_fields = ['game', 'levels']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field in config: {field}")

    if not isinstance(config['levels'], list):
        raise ValueError("'levels' must be a list")

    # Validate each level
    for i, level_spec in enumerate(config['levels']):
        if not isinstance(level_spec, dict):
            raise ValueError(f"Level {i} must be a dict")

        if 'level' not in level_spec:
            raise ValueError(f"Level {i} missing 'level' field")

        if 'transfer' in level_spec:
            transfer = level_spec['transfer']
            if not isinstance(transfer, dict):
                raise ValueError(f"Level {i} 'transfer' must be a dict")

            if 'from_level' not in transfer:
                raise ValueError(f"Level {i} 'transfer' missing 'from_level'")

            # Validate domain/predicates values - can be bool or "extend"
            for field in ['domain', 'predicates']:
                val = transfer.get(field)
                if val is not None and val not in [True, False, 'extend']:
                    raise ValueError(
                        f"Level {i} 'transfer.{field}' must be true, false, or 'extend', got: {val}"
                    )

    return config


def run_sequential_levels_with_transfer(
    agent,
    config: dict,
    build_engine_fn: Callable[[str, str, int], Any]
) -> List[Dict[str, Any]]:
    """Execute levels sequentially with configurable artifact transfer.

    Args:
        agent: TheoryCoderAgent instance
        config: Parsed YAML config with level and transfer specifications
        build_engine_fn: Function to build game engine: build_engine_fn(game, level_set, level_id)

    Returns:
        List of dicts with results for each level
    """
    game = config['game']
    levels_config = config['levels']
    max_attempts = config.get('max_attempts', 6)

    results = []

    for level_spec in levels_config:
        level_id = level_spec['level']
        transfer = level_spec.get('transfer', None)

        print(f"\n{'='*70}")
        print(f"[LEVEL {level_id}] Starting")
        if transfer:
            print(f"[TRANSFER] Will transfer from level {transfer['from_level']}")
        else:
            print(f"[LEARNING] No transfer - learning from scratch")
        print(f"{'='*70}")

        # Setup artifacts for this level
        if transfer:
            from_level = transfer['from_level']
            agent._setup_level_artifacts(level_id, from_level, transfer, game_name=game)
        else:
            # No transfer - just create artifact snapshot dir
            agent._setup_level_artifacts(level_id, None, None, game_name=game)

        # If a fixed worldmodel is configured, copy it into game_dir for every level
        fixed_wm_path = getattr(agent, 'fixed_worldmodel_path', None)
        if fixed_wm_path:
            fixed_wm = Path(fixed_wm_path)
            if fixed_wm.exists():
                dest_wm = agent.game_dir / 'worldmodel.py'
                shutil.copy(fixed_wm, dest_wm)
                print(f"[FIXED WM] Copied fixed worldmodel to {dest_wm}")
            else:
                print(f"[FIXED WM ERROR] Fixed worldmodel not found: {fixed_wm}")

        # Build engine for this level
        level_set_name = game
        engine = build_engine_fn(game, level_set_name, level_id)

        # Run agent for this level
        try:
            success = agent.run(engine, max_attempts=max_attempts)
        except Exception as e:
            print(f"[ERROR] Exception while running level {level_id}: {e}")
            import traceback
            traceback.print_exc()
            success = False

        # Snapshot artifacts after completion
        level_snapshot_dir = agent.game_dir / f'level_{level_id}_artifacts'
        try:
            agent._snapshot_artifacts_to_dir(level_snapshot_dir)
        except Exception as e:
            print(f"[WARNING] Failed to snapshot artifacts: {e}")

        # Record result
        result = {
            'level': level_id,
            'success': success,
            'transferred_from': transfer['from_level'] if transfer else None
        }
        results.append(result)

        print(f"\n[LEVEL {level_id}] {'✓ SUCCESS' if success else '✗ FAILED'}")

    return results


def seed_game_from_previous(
    agent,
    src_experiment_dir: str,
    game: str,
    copy_plans: bool = True,
    src_game: Optional[str] = None
):
    """Copy prior artifacts into current experiment for seeded runs.

    If src_game is provided (cross-game seeding), files are taken from
    tc_game/<src_game>/ and renamed to match <game> for domain/plans.

    Args:
        agent: TheoryCoderAgent instance
        src_experiment_dir: Path to source experiment directory
        game: Target game name
        copy_plans: Whether to copy plans file
        src_game: Source game name (if different from target)
    """
    import importlib
    import os

    src_game = src_game or game

    src = Path(src_experiment_dir) / "tc_game" / src_game
    dst = Path(agent.logger.experiment_dir) / "tc_game" / game
    dst.mkdir(parents=True, exist_ok=True)

    copies = [
        ("worldmodel.py", "worldmodel.py"),
        ("predicates.py", "predicates.py"),
        (f"{src_game}_domain.pddl", f"{game}_domain.pddl"),
    ]
    if copy_plans:
        copies.append((f"{src_game}_plans.json", f"{game}_plans.json"))

    for sname, dname in copies:
        sp = src / sname
        dp = dst / dname
        if sp.exists():
            shutil.copy(sp, dp)
            print(f"[seed] copied {sp} → {dp}")
        else:
            print(f"[seed] skip missing {sp}")

    agent.game_dir = dst
    agent.domain_file = str(dst / f"{game}_domain.pddl")
    os.environ["TC_WORLDMODEL_FILE"] = str(dst / "worldmodel.py")
    os.environ["TC_PREDICATES_FILE"] = str(dst / "predicates.py")

    agent.predicates_empty = False
    agent.domain_empty = False
    if "predicates" in sys.modules:
        importlib.reload(sys.modules["predicates"])
    agent.reload_predicates_module()

    print(f"[seed] seeded {game} from {src_experiment_dir}/tc_game/{src_game} → {dst}")
