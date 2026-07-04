"""PDDL parsing utilities."""
import re
from pathlib import Path
from typing import List, Tuple


def parse_sas_plan(path: str) -> List[str]:
    """Read a Fast-Downward sas_plan and return a list of action strings.

    Strips comments and parentheses from the plan file.

    Args:
        path: Path to the sas_plan file

    Returns:
        List of action strings
    """
    actions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(';'):
                continue
            actions.append(line.strip('()'))
    return actions


def extract_pddl_files(text: str) -> Tuple[str, str]:
    """Extract domain and problem code from LLM response text.

    Handles both the standard case where the LLM returns two separate
    ```pddl``` blocks and the degenerate case where both domain and
    problem appear within a single block.

    Args:
        text: Response text containing PDDL code blocks

    Returns:
        Tuple of (domain_code, problem_code)
    """
    pattern = r"```pddl(.*?)(?=```)"
    blocks = re.findall(pattern, text, re.DOTALL)
    domain, problem = "", ""

    if len(blocks) >= 2:
        domain = blocks[0].strip()
        problem = blocks[1].strip()
    elif len(blocks) == 1:
        combined = blocks[0].strip()
        dom_idx = combined.find("(define (domain")
        prob_idx = combined.find("(define (problem")
        if dom_idx != -1 and prob_idx != -1:
            if dom_idx < prob_idx:
                domain = combined[dom_idx:prob_idx].strip()
                problem = combined[prob_idx:].strip()
            else:
                problem = combined[prob_idx:dom_idx].strip()
                domain = combined[dom_idx:].strip()
        elif dom_idx != -1:
            domain = combined
        elif prob_idx != -1:
            problem = combined

    return domain, problem


def ensure_closed_parentheses(code: str) -> str:
    """Return code with any missing closing parentheses appended.

    Args:
        code: PDDL code string

    Returns:
        Code with balanced parentheses
    """
    depth = 0
    for ch in code:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
    if depth > 0:
        code += ')' * depth
    return code


def extract_code_block(text: str, lang: str, which: int = 1) -> str:
    """Extract a specific code block from text.

    Args:
        text: Text containing code blocks
        lang: Language identifier (e.g., 'python', 'pddl')
        which: 1-indexed block number to extract

    Returns:
        Extracted code block content
    """
    pattern = rf"```{lang}(.*?)(?=```)"
    blocks = re.findall(pattern, text, re.DOTALL)
    return blocks[which - 1].strip() if len(blocks) >= which else ""
