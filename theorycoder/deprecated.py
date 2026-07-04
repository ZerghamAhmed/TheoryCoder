"""Deprecated functions preserved for backwards compatibility.

These functions are superseded by the --transfer-config system.
They will be removed in a future release.

Deprecated CLI arguments:
- --babyai-sequence
- --babyai-reuse-strict
- --babyai-reuse-only

Use --transfer-config with a YAML file instead.
"""
import warnings
from typing import Tuple


def _deprecation_warning(name: str):
    """Issue a deprecation warning for a deprecated function."""
    warnings.warn(
        f"{name} is deprecated and will be removed in a future release. "
        f"Use --transfer-config with a YAML configuration file instead.",
        DeprecationWarning,
        stacklevel=3
    )


# These functions remain in theorycoder3.py for backwards compatibility
# but are not exposed in the theorycoder package.
#
# If you need to use them, import directly from theorycoder3:
#   from theorycoder3 import run_babyai_transfer_sequence
#
# But you should migrate to using --transfer-config instead.

DEPRECATED_CLI_ARGS = [
    "--babyai-sequence",
    "--babyai-reuse-strict",
    "--babyai-reuse-only",
    "--babyai-level-set",
]

MIGRATION_GUIDE = """
Migration Guide for Deprecated CLI Arguments
============================================

The following CLI arguments are deprecated and superseded by --transfer-config:

1. --babyai-sequence
   This ran levels 19 → 8 → 13 with hardcoded transfer logic.

   Replace with a YAML config like:

   ```yaml
   game: babyai
   experiment_dir: my_experiment
   learn_model: true
   levels:
     - level: 19
       # No transfer - learn from scratch
     - level: 8
       # No transfer - learn from scratch
     - level: 13
       transfer:
         from_level: 8
         worldmodel: true
         domain: true
         predicates: true
   ```

2. --babyai-reuse-strict
   This reused existing artifacts without any generation.

   Replace with a config using debug_no_llm: true

   ```yaml
   game: babyai
   debug_no_llm: true
   levels:
     - level: 19
     - level: 8
     - level: 13
   ```

3. --babyai-reuse-only
   Similar to --babyai-reuse-strict.

   Use the same debug_no_llm approach.

Example config files are in the configs/ directory.
"""


def print_migration_guide():
    """Print the migration guide for deprecated arguments."""
    print(MIGRATION_GUIDE)
