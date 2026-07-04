"""PDDL generation utilities.

This module contains functions for generating PDDL domain and problem files
using LLM prompts. The actual generation logic remains in the agent class,
but these utilities provide helper functions for the process.
"""
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from theorycoder.pddl.parser import extract_pddl_files, ensure_closed_parentheses


def save_pddl_files(
    domain_code: str,
    problem_code: str,
    domain_path: Path,
    problem_path: Path,
    step_dir: Optional[Path] = None,
) -> None:
    """Save PDDL domain and problem files.

    Args:
        domain_code: Domain PDDL code
        problem_code: Problem PDDL code
        domain_path: Path to save domain file
        problem_path: Path to save problem file
        step_dir: Optional step directory for logging copies
    """
    domain_code = ensure_closed_parentheses(domain_code)
    problem_code = ensure_closed_parentheses(problem_code)

    domain_path.write_text(domain_code)
    problem_path.write_text(problem_code)

    if step_dir:
        shutil.copy(str(domain_path), str(step_dir / domain_path.name))
        shutil.copy(str(problem_path), str(step_dir / problem_path.name))


def process_llm_pddl_response(
    response: str,
    domain_path: Path,
    problem_path: Path,
    step_dir: Optional[Path] = None,
) -> Tuple[str, str]:
    """Process LLM response containing PDDL code and save files.

    Args:
        response: LLM response text containing PDDL blocks
        domain_path: Path to save domain file
        problem_path: Path to save problem file
        step_dir: Optional step directory for logging

    Returns:
        Tuple of (domain_code, problem_code)
    """
    domain_code, problem_code = extract_pddl_files(response)
    save_pddl_files(domain_code, problem_code, domain_path, problem_path, step_dir)
    return domain_code, problem_code


def build_init_pddl_prompt(
    raw_state: str,
    mission: str,
    current_domain: str = "",
    prompt_loader=None,
    game_name: Optional[str] = None,
) -> str:
    """Build the initial PDDL generation prompt.

    Args:
        raw_state: JSON string of game state
        mission: Mission/goal description
        current_domain: Existing domain PDDL (if any)
        prompt_loader: Function to load prompt templates
        game_name: Optional game name for game-specific prompts

    Returns:
        Formatted prompt string
    """
    if prompt_loader is None:
        from theorycoder.llm_client import load_prompt
        prompt_loader = load_prompt

    return prompt_loader(
        "init_pddl_files",
        game_name=game_name,
        raw_state=raw_state,
        mission=mission,
        current_domain=current_domain,
    )


def build_transfer_problem_prompt(
    domain_file: str,
    raw_state: str,
    mission: str,
) -> str:
    """Build prompt for generating a problem file from existing domain.

    Args:
        domain_file: Content of existing domain file
        raw_state: JSON string of game state
        mission: Mission/goal description

    Returns:
        Formatted prompt string
    """
    tpl = Path("abstraction_prompts/transfer_domain.txt").read_text()

    try:
        prompt = tpl.format(
            domain_file=domain_file,
            raw_state=raw_state,
            mission=mission,
        )
    except KeyError:
        # Template doesn't have mission placeholder
        mission_header = "MISSION (context for this level):\n" + f"{mission}\n\n"
        prompt = mission_header + tpl.format(
            domain_file=domain_file,
            raw_state=raw_state,
        )

    return prompt


def build_regen_pddl_prompt(
    domain_code: str,
    problem_code: str,
    raw_state: str,
    current_domain: str = "",
) -> str:
    """Build prompt for regenerating failed PDDL.

    Args:
        domain_code: Current domain PDDL code
        problem_code: Current problem PDDL code
        raw_state: JSON string of game state
        current_domain: Full current domain (may include previous actions)

    Returns:
        Formatted prompt string
    """
    regen_tpl = Path("abstraction_prompts/regen_pddl_files.txt").read_text()

    regen_header = (
        "THIS IS THE CURRENT DOMAIN FILE BUILT SO FAR (IT CAN BE BLANK). "
        "ONLY EXTEND/EDIT THIS IF NEEDED; OTHERWISE REUSE IT.\n\n```pddl\n"
        f"{current_domain}\n```\n\n"
    )

    return regen_header + regen_tpl.format(
        domain_file=domain_code,
        problem_file=problem_code,
        raw_state=raw_state,
    )
