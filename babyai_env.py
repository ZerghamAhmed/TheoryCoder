# babyai_env.py

import gymnasium as gym
# import gym
from minigrid.wrappers import FullyObsWrapper
from babyai_lang_wrapper import MinigridTextObservationWrapper
from stateconvertutils import convert_minigrid_text_to_state
from copy import deepcopy

# Ensure custom MiniGrid environments are registered
# import single_key_env  # noqa: F401


BABYAI_LEVEL_MAPPING = {
    0: "BabyAI-GoToLocal-v0",
    1: "BabyAI-GoToRedBall-v0",
    2: "BabyAI-GoToObj-v0",
    3: "BabyAI-OpenRedDoor-v0",
    4: "BabyAI-OpenDoorDebug-v0",
    5: "BabyAI-OneRoomS20-v0",
    6: "BabyAI-PutNextLocal-v0",
    7: "BabyAI-MoveTwoAcrossS5N2-v0",
    8: "BabyAI-UnlockLocal-v0",
    9: "BabyAI-UnlockPickup-v0",
    10: "BabyAI-BlockedUnlockPickup-v0",
    11: "BabyAI-KeyInBox-v0",
    12: "BabyAI-PickupAbove-v0",
    13: "BabyAI-Unlock-v0",
    14: "BabyAI-MiniBossLevel-v0",
    15: "BabyAI-SynthSeq-v0",
    16: "BabyAI-Pickup-v0",
    17: "BabyAI-UnlockToUnlock-v0",
    18: "MiniGrid-Fetch-6x6-N2-v0",
}



class BabyAI:
    def __init__(self, level_set='babyai', level_id=0, seed=42):
        """
        Initialize the BabyAI environment.

        Args:
            env_name (str): Name of the BabyAI environment.
            seed (int): Random seed for reproducibility.
        """
        self.level_set = level_set
        self.level_id = level_id # baba had level id's so could remove this later
        self.env_name = BABYAI_LEVEL_MAPPING.get(level_id, None)
        
        # self.mission = 0
        self.mission = self._mission_for_level(level_id)


        self.seed = seed
        self.env = FullyObsWrapper(gym.make(self.env_name, render_mode='human'))
        self.env = MinigridTextObservationWrapper(self.env)
        self.actions_set = [
            "left", "right", "forward", "pickup", "drop", "toggle"
        ]
        # self.actions_set = [
        #    i for i in range(6)
        # ]
        # breakpoint()
        self.state_format = (
            "{\n"
            "    [(entity_name, x, y)],  # List of entities and their positions\n"
            "    'carrying': (item_name),  # Item being carried by the agent\n"
            "    'direction': <int>,  # Agent's direction\n"
            "    'won': <bool>,  # Whether the agent has won\n"
            "    'lost': <bool>   # Whether the agent has lost\n"
            "}"
        )
        self.reset()

    def _mission_for_level(self, lid: int) -> str:
        missions = {
            19: "pick up the key",
            13: "open the red door",
            8:  "open the door",
        }
        return missions.get(lid, "Reach the goal.")

    def reset(self):
        """
        Reset the environment and initialize the game state.
        """
        obs, _ = self.env.reset(seed=self.seed)
        self.text_obs = obs["text"]
        self.obs_carrying = obs["carrying"]
        self.obs_direction = obs["direction"]
        # self.mission = obs["mission"]
        self.mission = obs.get("mission") or self._mission_for_level(self.level_id)
        self.env_height = self.env.unwrapped.height
        self.env_width = self.env.unwrapped.width
        self.state = convert_minigrid_text_to_state(
            self.text_obs, self.env_height, self.env_width,
            self.obs_carrying, self.obs_direction
        )
        self.turn_number = 0
        self.won = False
        self.lost = False
    
    def step(self, action):
        """
        Execute an action in the environment.

        Args:
            action (str): Action to execute (e.g., "left", "forward").

        Returns:
            dict: Updated game state.
        """
        # print(f"Executing action: {action}")
        # if action not in self.actions_set:
        #     raise ValueError(f"Invalid action: {action}. Available actions: {self.actions_set}")
        action_idx = self.actions_set.index(action)
        obs, reward, done, _, info = self.env.step(action_idx)
        self.text_obs = obs["text"]
        self.obs_carrying = obs["carrying"]
        self.obs_direction = obs["direction"]
        # self.mission = obs["mission"]
        self.mission = obs.get("mission") or self._mission_for_level(self.level_id)

        self.state = convert_minigrid_text_to_state(
            self.text_obs, self.env_height, self.env_width,
            self.obs_carrying, self.obs_direction
        )
        self.turn_number += 1
        self.won = done

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
