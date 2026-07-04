"""Fast-Downward solver integration."""
import os
import subprocess
from pathlib import Path
from subprocess import CalledProcessError
from typing import List, Optional

import yaml


def load_downward_path(config_file: str = "downward_config.yaml") -> Optional[str]:
    """Return the Fast-Downward path from a YAML config if available.

    Args:
        config_file: Path to the config file

    Returns:
        Path to fast-downward.py or None
    """
    cfg_path = Path(config_file)
    if cfg_path.exists():
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if isinstance(data, dict):
                    return data.get("fast_downward_path")
        except Exception:
            pass
    return None


class FastDownwardSolver:
    """Wrapper for Fast-Downward PDDL solver."""

    def __init__(self, fast_downward_path: Optional[str] = None):
        """Initialize the solver.

        Args:
            fast_downward_path: Path to fast-downward.py. If not provided,
                will try environment variable, config file, then default.
        """
        self.path = (
            fast_downward_path
            or os.environ.get("FAST_DOWNWARD_PATH")
            or load_downward_path()
            or "fast-downward.py"
        )

    def solve(
        self,
        domain_path: Path,
        problem_path: Path,
        search: str = "astar(blind())",
        output_plan: str = "sas_plan",
        timeout: Optional[int] = None,
    ) -> bool:
        """Run Fast-Downward on the given domain and problem.

        Args:
            domain_path: Path to domain PDDL file
            problem_path: Path to problem PDDL file
            search: Search algorithm specification
            output_plan: Name of output plan file
            timeout: Optional timeout in seconds

        Returns:
            True if planning succeeded, False otherwise
        """
        cmd = [
            "python3", self.path,
            str(domain_path), str(problem_path),
            "--search", search,
        ]

        try:
            kwargs = {}
            if timeout is not None:
                kwargs["timeout"] = timeout

            subprocess.run(cmd, check=True, **kwargs)
            return True
        except CalledProcessError:
            return False
        except subprocess.TimeoutExpired:
            print(f"[FastDownward] Timeout after {timeout}s")
            return False

    def solve_and_parse(
        self,
        domain_path: Path,
        problem_path: Path,
        search: str = "astar(blind())",
        plan_file: str = "sas_plan",
    ) -> Optional[List[str]]:
        """Run Fast-Downward and parse the resulting plan.

        Args:
            domain_path: Path to domain PDDL file
            problem_path: Path to problem PDDL file
            search: Search algorithm specification
            plan_file: Name of output plan file

        Returns:
            List of action strings if successful, None otherwise
        """
        from theorycoder.pddl.parser import parse_sas_plan

        if self.solve(domain_path, problem_path, search):
            return parse_sas_plan(plan_file)
        return None
