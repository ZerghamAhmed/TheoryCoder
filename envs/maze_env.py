import sys
sys.path.insert(0, "/home/z/procgen")

# from procgen import ProcgenGym3Env
# from gym3 import ToGymEnv



MAZE_LEVEL_MAPPING = {
    0: "maze",  # only one level for now
}


class MazeEnv:
    def __init__(self, level_set='maze', level_id=0, seed=0, difficulty="easy"):
        """
        Initialize MazeEnv wrapper around a Procgen maze environment.
        """
        self.env_name = MAZE_LEVEL_MAPPING.get(level_id, "maze")
        self.seed = seed
        self.level_set = level_set
        self.level_id = level_id # baba had level id's so could remove this later

        # Optional: define a human-readable mapping
        self.actions_set = {
            0: "noop",
            1: "left",
            2: "right",
            3: "up",
            4: "down",
            # add more if you confirm more actions do something
        }


        self.env = ToGymEnv(ProcgenGym3Env(
            num=1,
            env_name=self.env_name,
            render_mode="rgb_array",
        ))

        self.reset()

    def reset(self):
        """
        Reset the environment and return symbolic state.
        """
        self.env.reset()
        self.done = False
        self.turn_number = 0
        return self._get_symbolic_state()

    def step(self, action):
        """
        Step the environment with a given action index (int).
        """
        _, reward, self.done, info = self.env.step(action)
        self.turn_number += 1
        return self._get_symbolic_state(), reward, self.done, info

    def _get_symbolic_state(self):
        """
        Extract symbolic representation: agent, goal, border.
        """
        info = self.env.env.get_info()[0]  # ✅ this is the raw info dict
        agent = [info["agent_x"], info["agent_y"]]
        goal = [info["goal_x"], info["goal_y"]]
        grid = info["walkable_grid"]  # shape (25, 25)

        border = [
            [x, y]
            for x in range(grid.shape[0])
            for y in range(grid.shape[1])
            if grid[x][y] == 0
        ]

        return {
            "avatar": agent,
            "goal": goal,
            "wall": border
        }

    def get_obs(self):
        """
        Get the current symbolic state.
        """
        return self._get_symbolic_state()

    def close(self):
        """
        Close the environment.
        """
        self.env.close()
