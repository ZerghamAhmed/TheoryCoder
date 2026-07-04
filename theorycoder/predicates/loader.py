"""Predicate loading utilities."""
import importlib
import sys
from pathlib import Path
from typing import Optional, Tuple


def load_predicates(
    predicates_file_name: str,
    exp_dir: Optional[Path] = None,
    base_dir: Optional[Path] = None,
) -> Tuple[Optional[Path], bool]:
    """Load predicates from the appropriate location.

    Search order:
    1. Experiment directory
    2. Base directory
    3. Current working directory

    Args:
        predicates_file_name: Name of the predicates module (without .py)
        exp_dir: Experiment directory path
        base_dir: Base directory path

    Returns:
        Tuple of (predicates_path, is_empty)
    """
    predicates_path = None

    # Search for predicates file
    if exp_dir:
        path = exp_dir / f"{predicates_file_name}.py"
        if path.exists():
            predicates_path = path

    if predicates_path is None and base_dir:
        path = base_dir / f"{predicates_file_name}.py"
        if path.exists():
            predicates_path = path

    if predicates_path is None:
        path = Path(f"{predicates_file_name}.py")
        if path.exists():
            predicates_path = path

    if predicates_path is None or not predicates_path.exists():
        print(f"Predicates file '{predicates_file_name}.py' not found.")
        return None, True

    # Check if empty
    with predicates_path.open('r') as f:
        content = f.read().strip()

    if not content:
        print(f"Warning: {predicates_file_name}.py is empty.")
        return predicates_path, True

    if "def " not in content:
        print(f"Warning: {predicates_file_name}.py has no function definitions.")
        return predicates_path, True

    return predicates_path, False


def is_predicates_empty(predicates_path: Optional[Path]) -> bool:
    """Check if predicates file is empty or doesn't exist.

    Args:
        predicates_path: Path to predicates file

    Returns:
        True if predicates are empty or missing
    """
    if predicates_path is None or not predicates_path.exists():
        return True

    content = predicates_path.read_text().strip()
    if not content or "def " not in content:
        return True

    return False


def reload_predicates_module() -> None:
    """Reload the predicates module if it's already loaded."""
    if "predicates" in sys.modules:
        try:
            importlib.reload(sys.modules["predicates"])
            print("[predicates] Reloaded predicates module")
        except Exception as e:
            print(f"[predicates] Failed to reload: {e}")
