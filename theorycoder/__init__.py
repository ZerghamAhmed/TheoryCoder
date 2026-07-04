"""TheoryCoder - Theory-based RL agent for game playing.

This package provides a modular implementation of the TheoryCoder agent,
which uses PDDL planning and world model learning to play games.

Usage:
    python -m theorycoder --transfer-config config.yaml

    Or via the legacy entry point:
    python theorycoder3.py --transfer-config config.yaml
"""
from theorycoder.config import AgentConfig, RuntimeState
from theorycoder.transfer import (
    load_transfer_config,
    run_sequential_levels_with_transfer,
    seed_game_from_previous,
)
from theorycoder.llm_client import LLMClient, load_prompt
from theorycoder.runner import LevelRunner, LevelConfig, LevelResult, DomainMode

__version__ = "0.1.0"

__all__ = [
    # Config
    "AgentConfig",
    "RuntimeState",
    # Transfer
    "load_transfer_config",
    "run_sequential_levels_with_transfer",
    "seed_game_from_previous",
    # LLM
    "LLMClient",
    "load_prompt",
    # Runner
    "LevelRunner",
    "LevelConfig",
    "LevelResult",
    "DomainMode",
]
