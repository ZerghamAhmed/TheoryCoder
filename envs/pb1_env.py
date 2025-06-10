from VGDLEnvAndres import VGDLEnvAndres
from stateconvertutils import *
from copy import deepcopy

class pb1env:
    def __init__(self, game_name="variant_expt_push_boulders_1_lvl0.txt", level_set="push_boulders_1", level_id=0, intended_steps=100000):
        """
        Initialize the push boulders 1 environment wrapper.

        Args:
            game_name (str): Name of the game file to load in VGDL.
            level_id (int): Starting level identifier (default is 0).
            seed (int): Random seed for reproducibility.
            intended_steps (int): Maximum number of steps allowed for the environment.
        """
        self.game_name = game_name
        self.level_id = level_id
        self.level_set = level_set
        self.intended_steps = intended_steps
        self.env = VGDLEnvAndres(game_name)

        # Initialize attributes for tracking the game state
        self.actions_set = ["noop", "right", "left", "up", "down"]
        self.won = False
        self.lost = False
        self.state = None
        self.turn_number = 0

        # Initialize the environment at the specified level
        self.set_level(level_id, intended_steps)
        self.reset()

    def set_level(self, level_id, intended_steps=None):
        """
        Set the current level in the environment.

        Args:
            level_id (int): Level identifier (e.g., 0 for lvl0, 1 for lvl1, etc.).
            intended_steps (int, optional): Maximum number of steps for the level.
        """
        self.level_id = level_id
        self.intended_steps = intended_steps or self.intended_steps
        self.env.set_level(self.level_id, self.intended_steps)

    def reset(self):
        """
        Reset the environment and initialize the game state.

        Returns:
            dict: Initial state of the environment.
        """
        self.env.reset()
        self.state = convert_pb1_state(self.env)
        self.state_colorized = convert_pb1_state_colorized(self.env)
        self.turn_number = 0
        self.won = False
        self.lost = False
        return deepcopy(self.state)

    def step(self, action):
        """
        Execute an action in the environment.

        Args:
            action (str): Action to execute (e.g., "left", "up", "noop").

        Returns:
            dict: Updated game state.
            float: Reward received.
            bool: Whether the game is over.
            dict: Additional info from the environment.
        """
        if action not in self.actions_set:
            raise ValueError(f"Invalid action: {action}. Available actions: {self.actions_set}")

        action_idx = self.actions_set.index(action)
        next_state, reward, done, info = self.env.step(action_idx)

        # Gracefully handle missing avatar (e.g., after death)
        try:
            self.state = convert_pb1_state(self.env, previous_state=self.state)
            self.state_colorized = convert_pb1_state_colorized(self.env, previous_state=self.state)
            # from planner import get_novelty_score, get_state_atoms
            # self.atoms = get_state_atoms(self.state)
            # print(self.atoms)
            self.save_screen()

        except AttributeError:
            print("Avatar is missing. Retaining previous state.")
            self.state = deepcopy(self.state)  # Keep the last known state

        self.turn_number += 1

        # Check win/lose conditions
        self._update_win_loss_conditions()

        return deepcopy(self.state), reward, done, info


    def _update_win_loss_conditions(self):
        """
        Update the `won` and `lost` attributes based on the current state.
        """
        if self.env.recent_history == [True]:  # Win condition
            self.won = True
            self.lost = False
        elif self.env.recent_history == [False]:  # Loss condition
            self.won = False
            self.lost = True
        else:  # Game is ongoing
            self.won = False
            self.lost = False

    def render(self):
        """
        Render the current state of the environment (if supported by VGDL).
        """
        self.env.render()

    def save_screen(self, filename="screenshot.png"):
        """
        Save the current screen as an image file.

        Args:
            filename (str): File name to save the screenshot.
        """
        self.env.save_screen(filename)

    def get_obs(self):
        """
        Get the current state of the environment.

        Returns:
            dict: Current game state.
        """
        return deepcopy(self.state)

    def close(self):
        """
        Close the environment.
        """
        self.env.close()
