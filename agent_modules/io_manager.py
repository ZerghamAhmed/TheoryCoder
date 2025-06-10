# io_manager.py

import json
import os
from pathlib import Path
import importlib

import ast
import re


def extract_function_str(file_content):
    function_pattern = r'def\s+([^\(]+)\('
    matches = re.finditer(function_pattern, file_content)
    function_names = set(match.group(1).strip() for match in matches)
    return function_names


def extract_code_from_response(response):
    # Use a regular expression to extract the Python code within ```python ``` tags (case-insensitive)
    code_match = re.search(r'```python(.*?)```', response, re.DOTALL | re.IGNORECASE)
    if code_match:
        return code_match.group(1).strip()
    else:
        return None


def is_world_model_empty(path: str = "worldmodel.py") -> bool:
    """
    Check if the worldmodel.py file is empty or missing.
    """
    p = Path(path)
    if not p.exists():
        return True
    content = p.read_text()
    return not content.strip()

def load_world_model(module_path):
    """
    Dynamically load a Python module from file (e.g. 'worldmodel.py')
    and return the module object.
    """
    path = Path(module_path)
    if not path.exists():
        raise FileNotFoundError(f"{module_path} not found")
    spec = importlib.util.spec_from_file_location(path.stem, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_json(file_path, default=None):
    """Load JSON from file_path, returning default on failure."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return default
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON in {file_path}: {e}")

def save_json(data, file_path):
    """Save Python object as pretty-printed JSON, creating parent dirs."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def read_text(file_path):
    """Return contents of a text file, or '' if missing."""
    try:
        return Path(file_path).read_text()
    except FileNotFoundError:
        return ""

def write_text(text, file_path):
    """Write text to file, creating parent dirs."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)

def load_plans(plans_file_name):
    """Load plans.json as dict, or empty dict if missing."""
    return load_json(plans_file_name, default={})

def load_domain_pddl(domain_file_name):
    """Load PDDL domain file; return content or ''."""
    content = read_text(domain_file_name).strip()
    return content

def load_predicates(predicates_module_name):
    """Load predicates.py file; return content or ''."""
    file_name = f"{predicates_module_name}.py"
    content = read_text(file_name).strip()
    return content

def capture_world_model(path="worldmodel.py"):
    """Return the full text of your current worldmodel.py."""
    return read_text(path)

def overwrite_world_model(new_code, path="worldmodel.py"):
    """Overwrite worldmodel.py with new_code."""
    write_text(new_code, path)

def save_level_summary(stats, level_key, experiment_dir):
    """Save per-level summary.txt and statistics.json."""
    level_dir = Path(experiment_dir) / f"level_{level_key}"
    level_dir.mkdir(parents=True, exist_ok=True)

    summary = (
        f"Level: {level_key}\n"
        f"Status: {stats['status']}\n"
        f"Attempts: {stats['attempts']}\n"
        f"Revisions: {stats['revisions']}\n"
        f"Debugs: {stats['debugs']}\n"
        f"Explorations: {stats['explorations']}\n"
        f"Solution: {stats.get('first_letters','None')}\n"
    )
    write_text(summary, level_dir / "summary.txt")
    save_json(stats, level_dir / "statistics.json")

def save_overall_summary(results, experiment_dir):
    """Save overall experiment summary.txt and results.json."""
    summary = (
        f"Total Levels Attempted: {len(results['levels_completed']) + len(results['levels_failed'])}\n"
        f"Levels Completed: {len(results['levels_completed'])}\n"
        f"Levels Failed: {len(results['levels_failed'])}\n"
        f"Total Revisions: {results['total_revisions']}\n"
        f"Total Debugs: {results['total_debugs']}\n"
        f"Total Explorations: {results['total_explorations']}\n\n"
        f"Completed Levels: {', '.join(results['levels_completed'])}\n"
        f"Failed Levels: {', '.join(results['levels_failed'])}\n"
    )
    write_text(summary, Path(experiment_dir) / "summary.txt")
    save_json(results, Path(experiment_dir) / "results.json")
