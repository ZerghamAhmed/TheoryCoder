# babyai_env.py

import gymnasium as gym
# import gym
from minigrid.wrappers import FullyObsWrapper
from envs.babyai_lang_wrapper import MinigridTextObservationWrapper
from envs.stateconvertutils import convert_minigrid_text_to_state
from copy import deepcopy

# Ensure custom MiniGrid environments are registered
import envs.single_key_env as single_key_env  # noqa: F401


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
    19: "MiniGrid-SingleKey-5x5-v0",
    20: "MiniGrid-Empty-5x5",
    21: "MiniGrid-LavaCrossingS11N5-v0",
    22: "MiniGrid-FourRooms-v0",
    23: "MiniGrid-Unlock-v0",
}

# seed 42 trial 1
# seed 5 trial 2 for llm + A
# seed 152


def _normalize_doors_only(state: dict) -> dict:
    """
    Convert door entries from the old format:
        locked_red_door / closed_red_door / open_red_door -> list (or single) of positions
    into the new format:
        red_door_1: {"location": [x,y], "open": bool, "locked": bool}

    Leaves all non-door entries unchanged.
    """
    if not isinstance(state, dict):
        return state

    # Collect and remove old door keys, then add new numbered door entries.
    new_state = dict(state)
    door_items = []

    # Any key containing "_door" (but not things like "agent_direction") is treated as a door bucket.
    for k in list(new_state.keys()):
        if "_door" not in k:
            continue

        positions = new_state.pop(k)

        # Normalize positions into a list of [x, y]
        pos_list = []
        if positions is None:
            pos_list = []
        elif isinstance(positions, list):
            # Could be [x,y] or [[x,y], ...]
            if len(positions) == 2 and all(isinstance(v, int) for v in positions):
                pos_list = [positions]
            else:
                # assume it's already list of positions
                pos_list = positions
        else:
            # Unknown format; skip safely
            continue

        # Determine door attributes from key prefix
        # Expected patterns: locked_red_door, closed_red_door, open_red_door
        parts = k.split("_")
        status = parts[0] if len(parts) >= 3 else ""
        color = parts[1] if len(parts) >= 3 else "unknown"

        if status == "locked":
            open_flag, locked_flag = False, True
        elif status == "open":
            open_flag, locked_flag = True, False
        else:
            # treat everything else (including "closed") as closed+unlocked
            open_flag, locked_flag = False, False

        for p in pos_list:
            # Expect p to be [x, y]
            if isinstance(p, list) and len(p) == 2 and all(isinstance(v, int) for v in p):
                door_items.append((color, p, open_flag, locked_flag))

    # Add numbered doors per color (stable ordering: insertion order from scanning keys/positions)
    per_color_counts = {}
    for color, loc, open_flag, locked_flag in door_items:
        per_color_counts[color] = per_color_counts.get(color, 0) + 1
        idx = per_color_counts[color]
        new_key = f"{color}_door_{idx}"
        new_state[new_key] = {
            "location": loc,
            "open": open_flag,
            "locked": locked_flag,
        }

    return new_state
# seed 42 success
# seed 5 success
# seed 123 success
#220
class BabyAI:
    def __init__(self, level_set='babyai', level_id=0, seed=42, legacy_door_format=False):
        """
        Initialize the BabyAI environment.

        Args:
            env_name (str): Name of the BabyAI environment.
            seed (int): Random seed for reproducibility.
            legacy_door_format (bool): If True, skip _normalize_doors_only
                and expose doors as the old 3-bucket format
                (locked_<color>_door / closed_<color>_door / open_<color>_door,
                each a list of positions). If False (default), use the new
                dict-per-door format (<color>_door_<N>: {location, open, locked}).
        """
        self.level_set = level_set
        self.level_id = level_id # baba had level id's so could remove this later
        self.env_name = BABYAI_LEVEL_MAPPING.get(level_id, None)

        # self.mission = 0
        self.mission = self._mission_for_level(level_id)

        self.seed = seed
        self.legacy_door_format = legacy_door_format
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
        if not self.legacy_door_format:
            self.state = _normalize_doors_only(self.state)
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
        if not self.legacy_door_format:
            self.state = _normalize_doors_only(self.state)
        self.turn_number += 1
        self.won = done
        # Return a Gym-style 4-tuple so callers that unpack the result (e.g.
        # WorldCoder v2's collect_random_experiences) work. Callers that ignore
        # the return value remain unaffected.
        return deepcopy(self.state), float(reward), bool(done), info

    def get_obs(self):
        """
        Get the current state of the environment.

        Returns:
            dict: Current game state.
        """
        return deepcopy(self.state)

    def get_rgb_frame(self):
        """Return the current full-resolution RGB frame as a uint8 HxWx3 array.

        Uses MiniGrid's get_frame() so it works regardless of render_mode.
        Used to build per-attempt GIFs of the executed action sequence.
        """
        import numpy as np
        return np.asarray(self.env.unwrapped.get_frame(), dtype=np.uint8)

    def close(self):
        """
        Close the environment.
        """
        self.env.close()
