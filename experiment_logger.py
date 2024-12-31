import os
import json
from time import time, strftime, gmtime
from typing import Dict, Any, Optional

class ExperimentLogger:
    def __init__(self, base_dir: str, experiment_name: Optional[str] = None):
        """Initialize experiment logger with a structured directory system."""
        self.base_dir = base_dir
        self.experiment_name = experiment_name or f"experiment_{strftime('%Y%m%d_%H%M%S', gmtime())}"
        self.experiment_dir = os.path.join(base_dir, "runs", self.experiment_name)
        self.current_step = 0
        self.tape = []
        self._initialize_experiment_directory()

    def _initialize_experiment_directory(self):
        """Create the main experiment directory structure."""
        # Create main directories
        os.makedirs(os.path.join(self.experiment_dir, "tape"), exist_ok=True)

    def _get_next_step_number(self) -> str:
        """Get the next step number in format 'XX'."""
        self.current_step += 1
        return f"{self.current_step:02d}"

    def create_step(self, step_type: str) -> str:
        """Create a new step directory and return its path."""
        step_num = self._get_next_step_number()
        step_dir = os.path.join(self.experiment_dir, f"step_{step_num}_{step_type}")
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
