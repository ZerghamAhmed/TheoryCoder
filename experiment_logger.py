import os
import json
from time import time, strftime, gmtime
from typing import Dict, Any, Optional
import re

class ExperimentLogger:
    def __init__(self, base_dir: str, experiment_name: Optional[str] = None,
                 create_subdir: bool = True):
        """Initialize experiment logger with a structured directory system.

        Parameters
        ----------
        base_dir: str
            Base directory where experiment data should be stored.
        experiment_name: Optional[str]
            Name of the experiment subdirectory. If ``None`` and
            ``create_subdir`` is ``True`` a timestamped directory will be
            created.  When ``create_subdir`` is ``False`` ``base_dir`` is used
            directly and ``experiment_name`` is ignored.
        create_subdir: bool, default ``True``
            Whether to create a new subdirectory for this experiment.
        """
        self.base_dir = base_dir
        self.create_subdir = create_subdir

        if create_subdir:
            self.experiment_name = (
                experiment_name
                or f"experiment_{strftime('%Y%m%d_%H%M%S', gmtime())}"
            )
            self.experiment_dir = os.path.join(base_dir, self.experiment_name)
        else:
            self.experiment_name = os.path.basename(base_dir.rstrip(os.sep))
            self.experiment_dir = base_dir
        self.current_step = 0
        self.tape = []
        self._initialize_experiment_directory()
        if not create_subdir:
            self._load_existing_steps()

    def _initialize_experiment_directory(self):
        """Create the main experiment directory structure."""
        # Create main directories
        os.makedirs(os.path.join(self.experiment_dir, "steps"), exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, "tc_game"), exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, "tape"), exist_ok=True)

    def _load_existing_steps(self):
        """Set ``current_step`` based on existing step directories."""
        steps_dir = os.path.join(self.experiment_dir, "steps")
        max_step = 0
        if os.path.isdir(steps_dir):
            for entry in os.listdir(steps_dir):
                if os.path.isdir(os.path.join(steps_dir, entry)):
                    m = re.match(r"^(\d+)_", entry)
                    if m:
                        try:
                            step_num = int(m.group(1))
                            if step_num > max_step:
                                max_step = step_num
                        except ValueError:
                            pass
        self.current_step = max_step

    def _get_next_step_number(self) -> str:
        """Get the next step number in format 'XXX'."""
        self.current_step += 1
        return f"{self.current_step:03d}"

    def create_step(self, step_type: str) -> str:
        """Create a new step directory and return its path."""
        step_num = self._get_next_step_number()
        step_dir = os.path.join(self.experiment_dir, "steps", f"{step_num}_{step_type}")
        os.makedirs(step_dir, exist_ok=True)
        return step_dir

    def save_step_files(self, step_dir: str, prompt: str, response: str, artifact_content: str = None, artifact_name: str = None):
        """Save prompt, response, and optional artifact files for a step."""
        with open(os.path.join(step_dir, "prompt.txt"), "w", encoding="utf-8") as f:
            f.write(prompt)
        
        with open(os.path.join(step_dir, "response.txt"), "w", encoding="utf-8") as f:
            f.write(response)

        if artifact_content and artifact_name:
            with open(os.path.join(step_dir, artifact_name), "w", encoding="utf-8") as f:
                f.write(artifact_content)

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        return strftime("%Y-%m-%dT%H:%M:%SZ", gmtime())

    def add_to_tape(self, entry: Dict[str, Any]):
        """Add an entry to the experiment tape with timestamp."""
        entry["timestamp"] = self._get_timestamp()
        self.tape.append(entry)
        self._save_tape()

    def _save_tape(self):
        """Save the current tape to disk."""
        tape_path = os.path.join(self.experiment_dir, "tape", "tape.json")
        with open(tape_path, "w", encoding="utf-8") as f:
            json.dump(self.tape, f, indent=2)

    def save_summary(self, summary: str):
        """Save the experiment summary."""
        with open(os.path.join(self.experiment_dir, "summary.txt"), "w", encoding="utf-8") as f:
            f.write(summary)

    def save_actions(self, level_key: str, actions) -> None:
        """Save the list of actions for a specific level."""
        level_dir = os.path.join(self.experiment_dir, f"level_{level_key}")
        os.makedirs(level_dir, exist_ok=True)
        actions_path = os.path.join(level_dir, "actions.json")
        with open(actions_path, "w", encoding="utf-8") as f:
            json.dump(actions, f, indent=2)
