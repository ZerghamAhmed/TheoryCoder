"""Abstract base class for game environments."""
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, List, Tuple


class BaseGameEnvironment(ABC):
    """Abstract base class for all game environments.

    This class defines the common interface that all game environments must implement.
    It provides a consistent API for the TheoryCoder agent to interact with different
    games.

    Attributes:
        level_set: Name of the level set
        level_id: ID of the current level
        actions_set: List of valid action strings
        won: Whether the game has been won
        lost: Whether the game has been lost
        state: Current game state dictionary
        turn_number: Number of turns taken
    """

    level_set: str
    level_id: int
    actions_set: List[str]
    won: bool = False
    lost: bool = False
    state: Dict[str, Any] = None
    turn_number: int = 0

    @abstractmethod
    def reset(self) -> Dict[str, Any]:
        """Reset the environment and return the initial state.

        Returns:
            Initial state dictionary
        """
        pass

    @abstractmethod
    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, dict]:
        """Execute an action and return the new state.

        Args:
            action: Action string to execute

        Returns:
            Tuple of (state, reward, done, info)
        """
        pass

    @abstractmethod
    def get_obs(self) -> Dict[str, Any]:
        """Get the current observation/state.

        Returns:
            Current state dictionary (deep copy)
        """
        pass

    def close(self) -> None:
        """Clean up resources. Override if needed."""
        pass

    def render(self) -> None:
        """Render the environment. Override if needed."""
        pass

    def save_screen(self, filename: str = "screenshot.png") -> None:
        """Save a screenshot. Override if needed."""
        pass

    def _validate_action(self, action: str) -> None:
        """Validate that an action is in the action set.

        Args:
            action: Action to validate

        Raises:
            ValueError: If action is not valid
        """
        if action not in self.actions_set:
            raise ValueError(
                f"Invalid action: {action}. Available actions: {self.actions_set}"
            )

    def _action_to_index(self, action: str) -> int:
        """Convert action string to index.

        Args:
            action: Action string

        Returns:
            Index of the action in actions_set
        """
        return self.actions_set.index(action)
