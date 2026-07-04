"""Base class for VGDL-based game environments."""
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

from theorycoder.environments.base import BaseGameEnvironment


class VGDLEnvironment(BaseGameEnvironment):
    """Base class for VGDL (Video Game Description Language) environments.

    This class provides common functionality for games built on the VGDL engine,
    reducing code duplication across similar environments like Sokoban, Labyrinth,
    and Cheesemaze.

    Subclasses should override:
        - _get_default_game_name(): Return the default game file name
        - _convert_state(): Convert raw VGDL state to game-specific format
        - _convert_state_colorized(): Convert to colorized state (optional)
    """

    def __init__(
        self,
        game_name: str,
        level_set: str,
        level_id: int = 0,
        intended_steps: int = 100000,
    ):
        """Initialize the VGDL environment.

        Args:
            game_name: Name of the VGDL game file
            level_set: Name for the level set
            level_id: Starting level ID
            intended_steps: Maximum steps allowed
        """
        # Import here to avoid circular imports and allow environments
        # to be used without VGDL installed
        from VGDLEnvAndres import VGDLEnvAndres

        self.game_name = game_name
        self.level_id = level_id
        self.level_set = level_set
        self.intended_steps = intended_steps
        self.env = VGDLEnvAndres(game_name)

        # Standard VGDL action set
        self.actions_set = ["noop", "right", "left", "up", "down"]
        self.won = False
        self.lost = False
        self.state = None
        self.state_colorized = None
        self.turn_number = 0

        # Initialize level
        self.set_level(level_id, intended_steps)
        self.reset()

    def set_level(self, level_id: int, intended_steps: Optional[int] = None) -> None:
        """Set the current level.

        Args:
            level_id: Level ID to set
            intended_steps: Maximum steps (uses current if None)
        """
        self.level_id = level_id
        self.intended_steps = intended_steps or self.intended_steps
        self.env.set_level(self.level_id, self.intended_steps)

    def reset(self) -> Dict[str, Any]:
        """Reset the environment."""
        self.env.reset()
        self.state = self._convert_state(self.env)
        self.state_colorized = self._convert_state_colorized(self.env)
        self.turn_number = 0
        self.won = False
        self.lost = False
        return deepcopy(self.state)

    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, dict]:
        """Execute an action."""
        self._validate_action(action)
        action_idx = self._action_to_index(action)
        next_state, reward, done, info = self.env.step(action_idx)

        # Handle missing avatar gracefully
        try:
            self.state = self._convert_state(self.env, previous_state=self.state)
            self.state_colorized = self._convert_state_colorized(
                self.env, previous_state=self.state
            )
            self.save_screen()
        except AttributeError:
            print("Avatar is missing. Retaining previous state.")
            self.state = deepcopy(self.state)

        self.turn_number += 1
        self._update_win_loss_conditions()

        return deepcopy(self.state), reward, done, info

    def _update_win_loss_conditions(self) -> None:
        """Update won/lost attributes based on VGDL history."""
        if self.env.recent_history == [True]:
            self.won = True
            self.lost = False
        elif self.env.recent_history == [False]:
            self.won = False
            self.lost = True
        else:
            self.won = False
            self.lost = False

    def get_obs(self) -> Dict[str, Any]:
        """Get current state."""
        return deepcopy(self.state)

    def render(self) -> None:
        """Render the environment."""
        self.env.render()

    def save_screen(self, filename: str = "screenshot.png") -> None:
        """Save screenshot."""
        self.env.save_screen(filename)

    def close(self) -> None:
        """Close the environment."""
        self.env.close()

    def _convert_state(
        self, env, previous_state: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Convert VGDL state to game-specific format.

        Override in subclasses for game-specific state conversion.

        Args:
            env: The VGDL environment
            previous_state: Previous state for incremental updates

        Returns:
            Converted state dictionary
        """
        from stateconvertutils import convert_pb1_state
        return convert_pb1_state(env, previous_state=previous_state)

    def _convert_state_colorized(
        self, env, previous_state: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Convert VGDL state to colorized format.

        Override in subclasses for game-specific conversion.

        Args:
            env: The VGDL environment
            previous_state: Previous state for incremental updates

        Returns:
            Colorized state dictionary
        """
        from stateconvertutils import convert_pb1_state_colorized
        return convert_pb1_state_colorized(env, previous_state=previous_state)


class RenamedVGDLEnvironment(VGDLEnvironment):
    """VGDL environment with entity renaming support.

    Subclasses can override _rename_entities() to rename state keys
    for better domain readability.
    """

    def reset(self) -> Dict[str, Any]:
        """Reset with entity renaming."""
        self.env.reset()
        self.state = self._rename_entities(self._convert_state(self.env))
        self.state_colorized = self._rename_entities(
            self._convert_state_colorized(self.env)
        )
        self.turn_number = 0
        self.won = False
        self.lost = False
        return deepcopy(self.state)

    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, dict]:
        """Step with entity renaming."""
        self._validate_action(action)
        action_idx = self._action_to_index(action)
        next_state, reward, done, info = self.env.step(action_idx)

        try:
            self.state = self._rename_entities(
                self._convert_state(self.env, previous_state=self.state)
            )
            self.state_colorized = self._rename_entities(
                self._convert_state_colorized(self.env, previous_state=self.state)
            )
            self.save_screen()
        except AttributeError:
            print("Avatar is missing. Retaining previous state.")
            self.state = deepcopy(self.state)

        self.turn_number += 1
        self._update_win_loss_conditions()

        return deepcopy(self.state), reward, done, info

    def _rename_entities(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Rename entities in the state.

        Override in subclasses to customize renaming.

        Args:
            state: State dictionary to rename

        Returns:
            State with renamed keys
        """
        return state
