from gymnasium.envs.registration import register
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.world_object import Key
from minigrid.core.mission import MissionSpace
from minigrid.core.grid import Grid


class SingleKeyEnv(MiniGridEnv):
    """A simple environment with one key to pick up."""

    def __init__(self, size: int = 5, max_steps: int | None = None, **kwargs) -> None:
        mission_space = MissionSpace(mission_func=self._gen_mission)
        if max_steps is None:
            max_steps = 4 * size ** 2
        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission() -> str:
        return "pick up the key"

    def _gen_grid(self, width: int, height: int) -> None:
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        self.place_agent()

        self.key = Key("yellow")
        self.place_obj(self.key)

        self.mission = self._gen_mission()

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        if action == self.actions.pickup and self.carrying is not None:
            if self.carrying == self.key:
                reward = self._reward()
                terminated = True

        return obs, reward, terminated, truncated, info


# Register the environment with gymnasium so ``gym.make`` can instantiate it.
register(
    id="MiniGrid-SingleKey-5x5-v0",
    entry_point="single_key_env:SingleKeyEnv",
    kwargs={"size": 5},
)
