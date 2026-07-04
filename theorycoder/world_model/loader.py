"""World model loading utilities."""
import importlib.util
import inspect
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def load_world_model(
    game_dir: Path,
    world_model_name: str = "worldmodel"
) -> Tuple[Any, bool]:
    """Load the world model from the game directory.

    Args:
        game_dir: Path to the game directory
        world_model_name: Name of the world model module

    Returns:
        Tuple of (world_model_module, is_empty)
    """
    model_path = game_dir / f"{world_model_name}.py"

    if not model_path.exists():
        print(f"World model file '{world_model_name}.py' not found in {game_dir}.")
        return None, True

    try:
        spec = importlib.util.spec_from_file_location("world_model", model_path)
        world_model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(world_model)
    except Exception as e:
        print(f"Error loading world model: {e}")
        return None, True

    is_empty = _check_world_model_empty(world_model)
    return world_model, is_empty


def _check_world_model_empty(world_model: Any) -> bool:
    """Check if a world model is effectively empty.

    Args:
        world_model: The loaded world model module

    Returns:
        True if the model is empty or a placeholder
    """
    if not hasattr(world_model, 'transition_model'):
        print("Warning: transition_model function not found.")
        return True

    try:
        transition_model_code = inspect.getsource(world_model.transition_model).strip()

        placeholder_code = "def transition_model(state, action):\n    return state"

        if transition_model_code == placeholder_code:
            print("Warning: transition_model is unimplemented (placeholder).")
            return True
        elif len(transition_model_code.splitlines()) <= 2:
            print("Warning: transition_model is effectively empty.")
            return True

        return False

    except Exception as e:
        print(f"Error checking world model: {e}")
        return True


def is_world_model_empty(runtime_vars: Dict[str, Any]) -> bool:
    """Check if the world model is empty based on runtime vars.

    Args:
        runtime_vars: Runtime variables dictionary

    Returns:
        True if the world model is empty
    """
    wm_str = runtime_vars.get('world_model_str', '')
    if not wm_str.strip():
        return True

    # Check for placeholder patterns
    placeholder_patterns = [
        "def transition_model(state, action):\n    return state",
        "def transition_model(state, action): return state",
    ]

    normalized = ' '.join(wm_str.split())
    for pattern in placeholder_patterns:
        if ' '.join(pattern.split()) in normalized:
            return True

    # Check if it's too short to be meaningful
    lines = [l for l in wm_str.strip().splitlines() if l.strip() and not l.strip().startswith('#')]
    if len(lines) <= 3:
        return True

    return False


def capture_world_model(game_dir: Path) -> str:
    """Load and return the world model code as a string.

    Creates a placeholder if the file doesn't exist.

    Args:
        game_dir: Path to the game directory

    Returns:
        World model code as string
    """
    world_model_path = game_dir / "worldmodel.py"

    if not world_model_path.exists():
        # Minimal placeholder to allow execution to continue
        world_model_str = (
            "from utils import directions\n\n"
            "def transition_model(state, action):\n"
            "    return state\n"
        )
        world_model_path.write_text(world_model_str)
    else:
        world_model_str = world_model_path.read_text()

    return world_model_str


def overwrite_world_model(game_dir: Path, new_code: str) -> None:
    """Overwrite the world model file with new code.

    Args:
        game_dir: Path to the game directory
        new_code: New world model code
    """
    world_model_path = game_dir / "worldmodel.py"
    world_model_path.write_text(new_code)
    print(f"[world_model] Wrote new model to {world_model_path}")
