from envs.VGDLEnvAndres import VGDLEnvAndres
from envs.stateconvertutils import *
from copy import deepcopy


class CheesemazeEnv:
    def __init__(self, game_name="c1movetogame2_lvl0.txt", level_set="cheesemaze", level_id=0, intended_steps=100000):
        """
        Initialize the Cheesemaze environment wrapper.

        Args:
            game_name (str): Name of the VGDL game file to load.
            level_set (str): Environment name for logging or saving.
            level_id (int): Level index.
            intended_steps (int): Max number of steps allowed.
        """
        self.game_name = game_name
        self.level_id = level_id
        self.level_set = level_set
        self.intended_steps = intended_steps
        self.env = VGDLEnvAndres(game_name)

        self.actions_set = ["noop", "right", "left", "up", "down"]
        self.won = False
        self.lost = False
        self.state = None
        self.turn_number = 0

        self.set_level(level_id, intended_steps)
        self.reset()

    # ------------------------------------------------------------
    # Utility to rename "avatar"→"rat" and "goal"→"cheese"
    # ------------------------------------------------------------
    def _rename_entities(self, state):
        """Rename entities in the state for domain readability."""
        new_state = {}
        for k, v in state.items():
            if k == "avatar":
                new_state["rat"] = v
            elif k == "goal":
                new_state["cheese"] = v
            else:
                new_state[k] = v
        return new_state

    # ------------------------------------------------------------
    def set_level(self, level_id, intended_steps=None):
        """Set the current level in the environment."""
        self.level_id = level_id
        self.intended_steps = intended_steps or self.intended_steps
        self.env.set_level(self.level_id, self.intended_steps)

    def reset(self):
        """Reset the environment and initialize the game state."""
        self.env.reset()
        self.state = self._rename_entities(convert_pb1_state(self.env))
        self.state_colorized = self._rename_entities(convert_pb1_state_colorized(self.env))
        self.turn_number = 0
        self.won = False
        self.lost = False
        return deepcopy(self.state)

    def step(self, action):
        """Execute an action in the environment."""
        if action not in self.actions_set:
            raise ValueError(f"Invalid action: {action}. Available: {self.actions_set}")

        action_idx = self.actions_set.index(action)
        next_state, reward, done, info = self.env.step(action_idx)

        try:
            self.state = self._rename_entities(convert_pb1_state(self.env, previous_state=self.state))
            self.state_colorized = self._rename_entities(
                convert_pb1_state_colorized(self.env, previous_state=self.state)
            )
            self.save_screen()
        except AttributeError:
            print("Rat is missing (avatar not found). Retaining previous state.")
            self.state = deepcopy(self.state)

        self.turn_number += 1
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
        """Render the current state of the environment."""
        self.env.render()

    def save_screen(self, filename="screenshot.png"):
        """Save the current screen as an image file."""
        self.env.save_screen(filename)

    def get_obs(self):
        """Return the current state."""
        return deepcopy(self.state)

    def close(self):
        """Close the environment."""
        self.env.close()


# ------------------------------------------------------------
# Example usage
# ------------------------------------------------------------
# if __name__ == "__main__":
#     env = CheesemazeEnv()
#     obs = env.reset()
#     print("Initial state keys:", obs.keys())  # Should show "rat" and "cheese"

#     for step in range(5):
#         obs, reward, done, info = env.step("right")
#         print(f"Step {step}: reward={reward}, done={done}")
#         if done:
#             break

#     env.close()
