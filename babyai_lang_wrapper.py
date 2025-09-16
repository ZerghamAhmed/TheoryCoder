from copy import deepcopy
from gymnasium import spaces
from gymnasium.core import ObservationWrapper, ActionWrapper

from minigrid.core.actions import Actions
from minigrid.core.constants import IDX_TO_OBJECT, IDX_TO_COLOR, STATE_TO_IDX

IDX_TO_STATE = {v: k for k, v in STATE_TO_IDX.items()}

DIRECTIONS2VEC = {
    0: (1, 0),  # Right
    1: (0, -1),  # Down
    2: (-1, 0), # Left
    3: (0, 1)  # Up
}


def get_inventory(env):
    """Extracts inventory details from the environment."""
    carrying = env.unwrapped.carrying  # Access the carrying object
    if carrying:
        return {
            "type": carrying.type,   # Use the object's type directly
            "color": carrying.color  # Use the object's color directly
        }
    return {"type": "none", "color": "none"}


class MinigridTextObservationWrapper(ObservationWrapper):
    @property
    def spec(self):
        return self.env.spec

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = deepcopy(self.env.observation_space)
        self.observation_space.spaces["text"] = spaces.Text(max_length=4096)
        self.observation_space.spaces["direction"] = spaces.Box(low=-1, high=1, shape=(2,), dtype="int")
        self.observation_space.spaces["carrying"] = spaces.Dict({
            "type": spaces.Text(max_length=16),
            "color": spaces.Text(max_length=16)
        })
        self.observation_space.spaces["x-position"] = spaces.Discrete(env.unwrapped.width)
        self.observation_space.spaces["y-position"] = spaces.Discrete(env.unwrapped.height)

    def observation(self, observation) -> dict:
        # Decode objects in the 'image' observation
        text_obs = ""
        image = self.unwrapped.grid.encode()  # Fetch updated grid
        h, w, _ = image.shape
        for i in range(h):
            for j in range(w):
                cell = image[i, j]
                object_idx, color_idx, state_idx = cell
                object_type = IDX_TO_OBJECT[object_idx]
                color = IDX_TO_COLOR[color_idx]

                if object_type == "door":
                    state = IDX_TO_STATE[state_idx] + " "
                else:
                    state = ""

                if object_type == "unseen" or object_type == "empty":
                    continue
                else:
                    text_obs += f"{state}{color} {object_type} at {(i, j)}\n"

        # Add the agent explicitly
        agent_x, agent_y = map(int, self.unwrapped.agent_pos)  # Convert to plain integers
        agent_color = "red"  # Default agent color in MiniGrid
        text_obs += f"{agent_color} agent at ({agent_x}, {agent_y})\n"

        # Add agent direction
        direction = DIRECTIONS2VEC[self.unwrapped.agent_dir]
        observation["direction"] = direction

        # Add inventory information
        observation["carrying"] = get_inventory(self.unwrapped)

        # Add agent position
        observation["x-position"], observation["y-position"] = agent_x, agent_y

        # Add text observation
        observation["text"] = text_obs
        return observation


class MinigridTextActionWrapper(ActionWrapper):
    @property
    def spec(self):
        return self.env.spec

    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Text(max_length=8)

    def action(self, action) -> Actions:
        action_str = action.strip().lower()
        if action_str == "left":
            return Actions.left
        elif action_str == "right":
            return Actions.right
        elif action_str == "forward":
            return Actions.forward
        elif action_str == "pickup":
            return Actions.pickup
        elif action_str == "drop":
            return Actions.drop
        elif action_str == "toggle":
            return Actions.toggle
        elif action_str == "done":
            return Actions.done
        else:
            raise ValueError(f"Unrecognized action: {action_str}")
