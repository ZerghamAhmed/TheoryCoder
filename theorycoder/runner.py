"""Level runner for TheoryCoder agent.

This module provides a structured interface for running levels with the agent.
The run() logic is complex and tightly coupled to agent state, so this module
provides helper classes and utilities to support incremental refactoring.
"""
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class DomainMode(Enum):
    """Domain handling mode for a level."""
    DEBUG = auto()      # Debug mode - skip LLM calls
    EXTEND = auto()     # Extend existing domain with new operator
    TRANSFER = auto()   # Transfer domain as-is, generate new problem
    SCRATCH = auto()    # Generate domain from scratch
    EXISTING = auto()   # Use existing domain and problem


@dataclass
class LevelConfig:
    """Configuration for running a level."""
    level_id: int
    max_attempts: int = 6
    max_revisions: int = 5

    # Transfer settings
    transfer_from: Optional[int] = None
    transfer_worldmodel: bool = False
    transfer_domain: bool = False  # True, False, or 'extend'
    transfer_predicates: bool = False  # True, False, or 'extend'

    # Debug settings
    debug_no_llm: bool = False
    planner_timeout: Optional[int] = None


@dataclass
class LevelResult:
    """Result from running a level."""
    level_id: int
    success: bool
    attempts: int = 0
    revisions: int = 0
    debugs: int = 0
    explorations: int = 0
    first_letters: Optional[str] = None
    error: Optional[str] = None


@dataclass
class LevelStatistics:
    """Statistics tracked during level execution."""
    attempts: int = 0
    revisions: int = 0
    debugs: int = 0
    explorations: int = 0
    status: str = "in_progress"
    first_letters: Optional[str] = None


def determine_domain_mode(
    domain_path: Path,
    problem_path: Path,
    transfer_config: Optional[dict],
    debug_no_llm: bool,
) -> DomainMode:
    """Determine which domain handling mode to use.

    Args:
        domain_path: Path to domain PDDL file
        problem_path: Path to problem PDDL file
        transfer_config: Transfer configuration dict (if any)
        debug_no_llm: Whether debug mode is enabled

    Returns:
        The DomainMode to use
    """
    if debug_no_llm:
        return DomainMode.DEBUG

    if transfer_config:
        domain_mode = transfer_config.get('domain', False)
        if domain_mode == 'extend':
            return DomainMode.EXTEND
        elif domain_mode is True:
            return DomainMode.TRANSFER

    if not domain_path.exists():
        return DomainMode.SCRATCH

    return DomainMode.EXISTING


class LevelRunner:
    """Runner for individual levels.

    This class encapsulates the level execution logic, providing a cleaner
    interface than the monolithic run() method while maintaining compatibility
    with existing agent code.

    Note: This is a refactoring target. Currently it mostly delegates to
    agent methods, but over time more logic can be moved here.
    """

    def __init__(self, agent):
        """Initialize the runner.

        Args:
            agent: TheoryCoderAgent instance
        """
        self.agent = agent
        self.stats = LevelStatistics()

    def setup_game_directory(self, game_name: str) -> Path:
        """Setup the game directory for a level.

        Args:
            game_name: Name of the game

        Returns:
            Path to the game directory
        """
        import os
        import sys

        game_dir = Path(self.agent.logger.experiment_dir) / "tc_game" / game_name
        game_dir.mkdir(parents=True, exist_ok=True)

        # Ensure planner loads the correct world model
        os.environ["TC_WORLDMODEL_FILE"] = str(game_dir / "worldmodel.py")
        if str(game_dir) not in sys.path:
            sys.path.insert(0, str(game_dir))
        if self.agent.logger.experiment_dir not in sys.path:
            sys.path.insert(0, self.agent.logger.experiment_dir)

        return game_dir

    def handle_domain_mode(
        self,
        mode: DomainMode,
        level: int,
        domain_path: Path,
        problem_path: Path,
    ) -> Optional[List[str]]:
        """Handle domain generation based on mode.

        Args:
            mode: The domain mode to use
            level: Current level ID
            domain_path: Path to domain file
            problem_path: Path to problem file

        Returns:
            Plan actions if successful, None otherwise
        """
        if mode == DomainMode.DEBUG:
            return self._handle_debug_mode(level, domain_path, problem_path)
        elif mode == DomainMode.EXTEND:
            return self._handle_extend_mode(level, domain_path)
        elif mode == DomainMode.TRANSFER:
            return self._handle_transfer_mode(level, domain_path)
        elif mode == DomainMode.SCRATCH:
            return self._handle_scratch_mode(level, domain_path)
        else:
            return self._handle_existing_mode(level, domain_path)

    def _handle_debug_mode(
        self, level: int, domain_path: Path, problem_path: Path
    ) -> Optional[List[str]]:
        """Handle debug mode - use existing artifacts without LLM calls."""
        import subprocess
        from subprocess import CalledProcessError

        game_name = self.agent._get_game_name()
        print(f"[{game_name}] DEBUG MODE: Skipping LLM calls, using existing artifacts")

        if domain_path.exists() and problem_path.exists():
            print(f"[{game_name}] Using existing domain and problem files")
            cmd = [
                "python3", self.agent.fast_downward_path,
                str(domain_path), str(problem_path),
                "--search", "astar(blind())",
            ]
            try:
                subprocess.run(cmd, check=True)
                plan = self.agent.parse_sas_plan("sas_plan")
                self.agent._save_plan_for_problem(problem_path, plan)
                return plan
            except CalledProcessError as e:
                print(f"[DEBUG MODE] Fast Downward failed: {e.returncode}")
                return None
        else:
            print(f"[{game_name}] ERROR: Missing domain or problem file for debug mode")
            return None

    def _handle_extend_mode(self, level: int, domain_path: Path) -> Optional[List[str]]:
        """Handle extend mode - extend domain with new operator."""
        game_name = self.agent._get_game_name()
        print(f"[{game_name}] EXTEND MODE: Extending domain with new operator")

        if not domain_path.exists():
            print(f"[{game_name}] WARNING: Domain transfer failed, generating from scratch")
            return self.agent.generate_and_solve_pddl(level=level)

        return self.agent.extend_domain_and_generate_problem(level=level)

    def _handle_transfer_mode(self, level: int, domain_path: Path) -> Optional[List[str]]:
        """Handle transfer mode - use transferred domain, generate problem."""
        game_name = self.agent._get_game_name()
        print(f"[{game_name}] TRANSFER MODE: Using transferred domain, generating new problem")

        if not domain_path.exists():
            print(f"[{game_name}] WARNING: Domain transfer failed, generating from scratch")
            return self.agent.generate_and_solve_pddl(level=level)

        return self.agent.generate_problem_for_existing_domain(level=level)

    def _handle_scratch_mode(self, level: int, domain_path: Path) -> Optional[List[str]]:
        """Handle scratch mode - generate domain from scratch."""
        game_name = self.agent._get_game_name()
        print(f"[{game_name}] domain PDDL not found -> generating now")
        return self.agent.generate_and_solve_pddl(level=level)

    def _handle_existing_mode(self, level: int, domain_path: Path) -> Optional[List[str]]:
        """Handle existing mode - use existing domain and solve."""
        game_name = self.agent._get_game_name()
        print(f"[{game_name}] domain PDDL already exists, skipping generation")
        return self.agent._solve_existing_pddl(level, domain_path)


def run_level(
    agent,
    engine,
    config: LevelConfig,
) -> LevelResult:
    """Run a single level with the agent.

    This is a high-level interface that wraps the agent's run() method
    with structured configuration and results.

    Args:
        agent: TheoryCoderAgent instance
        engine: Game engine instance
        config: Level configuration

    Returns:
        LevelResult with execution details
    """
    try:
        success = agent.run(engine, max_attempts=config.max_attempts)

        # Extract statistics from agent
        level_key = f"{engine.level_set}_{config.level_id}"
        stats = agent.level_statistics.get(level_key, {})

        return LevelResult(
            level_id=config.level_id,
            success=success,
            attempts=stats.get('attempts', 0),
            revisions=stats.get('revisions', 0),
            debugs=stats.get('debugs', 0),
            explorations=stats.get('explorations', 0),
            first_letters=stats.get('first_letters'),
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return LevelResult(
            level_id=config.level_id,
            success=False,
            error=str(e),
        )
