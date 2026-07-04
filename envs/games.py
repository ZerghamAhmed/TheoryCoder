import numpy as np
import json
import subprocess
from pathlib import Path
from copy import deepcopy
from itertools import product


class BabaIsYou:
    # Assign ascii values to images
    map_key = {
        '_': "border",
        ' ': "empty",
        'b': "baba_obj",
        'B': "baba_word",
        '1': "is_word",
        '2': "you_word",
        '3': "win_word",
        's': "skull_obj",
        'S': "skull_word",
        'f': "flag_obj",
        'F': "flag_word",
        'o': "floor_obj",
        'O': "floor_word",
        'a': "grass_obj",
        'A': "grass_word",
        '4': "kill_word",
        'l': "lava_obj",
        'L': "lava_word",
        '5': "push_word",
        'r': "rock_obj",
        'R': "rock_word",
        '6': "stop_word",
        'w': "wall_obj",
        'W': "wall_word",
        '7': "move_word",
        '8': "hot_word",
        '9': "melt_word",
        'k': "keke_obj",
        'K': "keke_word",
        'g': "goop_obj",
        'G': "goop_word",
        '0': "sink_word",
        'v': "love_obj",
        'V': "love_word",
    }

    def __init__(
        self,
        level_set='demo_LEVELS',
        level_id=0,
        js_engine_path='KekeCompetition-main/Keke_JS/interface.js',
        intermediate_gamestates_dir='_baba_gamestates_tmp2',
        model_name='gpt-4o',
        temperature=1.0,
    ):
        # Set up for Baba Is You engine
        if not Path(intermediate_gamestates_dir).exists():
            Path(intermediate_gamestates_dir).mkdir(parents=True, exist_ok=True)
        self.js_engine_path = js_engine_path
        self.intermediate_gamestates_dir = intermediate_gamestates_dir
        self.level_set = level_set
        self.level_id = level_id

        # Standard setup for all games
        self.actions_set = ['up', 'down', 'left', 'right']
        self.state_format = (
            "\{\n"
            "    <object 1>: [(x, y)],  # some object class and its location\n"
            "    <object 2>: [(x, y), ...],  # some other object class and its locations\n"
            "    ...  # etc.\n"
            "    'lost': <bool>,  # Whether game has been lost yet\n"
            "    'won': <bool>,  # Whether game has been won yet\n"
            "\}"
        )
        self.reset()

    def _resolve_level_index(self, level_set, level_id):
        """
        Convert a level 'id' (as stored in JSON) to the correct array index.
        If level_id is already an array index, return it directly.
        """
        levels_path = Path("KekeCompetition-main/Keke_JS/json_levels") / f"{level_set}.json"
        if not levels_path.exists():
            raise FileNotFoundError(f"Level set file not found: {levels_path}")

        with open(levels_path, "r") as f:
            data = json.load(f)

        levels = data.get("levels", [])
        # First try to match by "id"
        for idx, lvl in enumerate(levels):
            if str(lvl.get("id")) == str(level_id):
                return idx

        # If not found, assume level_id was already an index
        if isinstance(level_id, int) and 0 <= level_id < len(levels):
            return level_id

        raise ValueError(f"Level '{level_id}' not found in {levels_path}")

    def game_engine(self, move, turn_number, level_set, level_id):
        if move == 'None':
            load_pth = 'None'
            save_pth = Path(self.intermediate_gamestates_dir).joinpath(
                f'init_state.json'
            )
        elif turn_number == 0:
            load_pth = 'None'
            save_pth = Path(self.intermediate_gamestates_dir).joinpath(
                f'turn_0.json'
            )
        else:
            load_pth = Path(self.intermediate_gamestates_dir).joinpath(
                f'turn_{turn_number - 1}.json'
            )
            save_pth = Path(self.intermediate_gamestates_dir).joinpath(
                f'turn_{turn_number}.json'
            )
        try:
            # resolve to array index before passing to Node
            level_index = self._resolve_level_index(level_set, level_id)
            stdout = subprocess.check_output(
                ['node', self.js_engine_path, str(load_pth), str(save_pth), level_set, str(level_index), move],
                text=True
            )
        except subprocess.CalledProcessError as e:
            print("Error:", e.output)
            raise Exception()

        with save_pth.open('r') as fid:
            state = json.load(fid)

        # Debugging output
        # print("Engine Output State:", state)
        return state

    def initialize_map(self, level_set, level_id):
        return self.game_engine('None', 0, level_set, level_id)

    def get_obj_coords(self, state):
        ns = state["state"]["next_state"]
        obj_map = ns["obj_map"]
        back_map = ns["back_map"]

        # Infer dimensions (obj_map is indexed as [x][y])
        cols = len(obj_map)              # width  (x)
        rows = len(obj_map[0]) if cols else 0  # height (y)

        def add(obs, name, coord):
            if name in obs:
                if coord not in obs[name]:
                    obs[name].append(coord)
            else:
                obs[name] = [coord]

        obs = {}
        for y in range(rows):
            for x in range(cols):
                cell = obj_map[x][y]
                back = back_map[x][y]

                if isinstance(cell, dict):
                    name = cell["name"] + ("_obj" if cell["type"] == "phys" else "_word")
                    cx, cy = cell["x"], rows - 1 - cell["y"]
                    add(obs, name, (cx, cy))
                elif isinstance(back, dict):
                    name = back["name"] + ("_obj" if back["type"] == "phys" else "_word")
                    cx, cy = back["x"], rows - 1 - back["y"]
                    add(obs, name, (cx, cy))
                else:
                    # background cells use single-character tokens ('_' or ' ')
                    if back == '_':
                        name = 'border'
                    elif back == ' ':
                        name = 'empty'
                    else:
                        # If your JS can emit other background tokens, map them here (or ignore).
                        # For now, treat unknowns as empty instead of crashing.
                        name = 'empty'
                    # Fallback to the grid coordinates for background tiles
                    cx, cy = x, rows - 1 - y
                    add(obs, name, (cx, cy))

        # Add any overlaps that weren't captured above
        for ov in ns.get("overlaps", []):
            name = ov["name"] + ("_obj" if ov["type"] == "phys" else "_word")
            cx, cy = ov["x"], rows - 1 - ov["y"]
            add(obs, name, (cx, cy))

        return obs

    def reset(self):
        self.turn_number = 0
        engine_out = self.initialize_map(self.level_set, self.level_id)
        self.state = self.get_obj_coords(engine_out)
        self.won = False
        self.lost = False

    def step(self, action):
        engine_out = self.game_engine(
            action, self.turn_number, self.level_set, self.level_id
        )
        self.state = self.get_obj_coords(engine_out)  # Returns {'state': ..., 'won': ...} after initialization
        self.won = engine_out['won']
        self.lost = not len(engine_out['state']['next_state']['players'])
        self.turn_number += 1

    def get_obs(self):
        state = deepcopy(self.state)
        state['won'] = self.won
        state['lost'] = self.lost
        return state


class LavaGrid:
    def __init__(self, bounds=((0, 4), (0, 4)), avatar_init=(0, 0), goal=(2, 2)):
        self.bounds = bounds
        self.avatar_init = avatar_init
        self.goal = goal
        self.actions_set = ['up', 'down', 'left', 'right']
        self.state_format = (
            "\{\n"
            "    'avatar': (x, y),  # player coordinate\n"
            "    'goal': (x, y),  # goal location\n"
            "    'red_squares': [(x1, y1), (x2, y2), ...],  # Which squares are red\n"
            "    'blue_squares': [(x1, y1), (x2, y2), ...],  # Which squares are blue\n"
            "    'lost': <bool>,  # Whether game has been lost yet\n"
            "    'won': <bool>,  # Whether game has been won yet\n"
            "\}"
        )
        self.reset()

    def reset(self):
        self.state = {
            'avatar': self.avatar_init,
            'goal': self.goal,
            'red_squares': [
                (x, y) for x, y in product(
                    range(self.bounds[0][0], self.bounds[0][1] + 1),
                    range(self.bounds[1][0], self.bounds[1][1] + 1)
                ) if x > y
            ],
            'blue_squares': [
                (x, y) for x, y in product(
                    range(self.bounds[0][0], self.bounds[0][1] + 1),
                    range(self.bounds[1][0], self.bounds[1][1] + 1)
                ) if not x > y
            ],
            'won': False,
            'lost': False,
        }
        self.won = False
        self.lost = False

    @staticmethod
    def check_win(state):
        if state['avatar'] == state['goal']:
            return True
        else:
            return False

    def step(self, action):
        state = deepcopy(self.state)
        if self.won:
            return
        if self.lost:
            return False
        if action == 'up':
            state['avatar'] = (state['avatar'][0], state['avatar'][1] + 1)
        if action == 'down':
            state['avatar'] = (state['avatar'][0], state['avatar'][1] - 1)
        if action == 'left':
            state['avatar'] = (state['avatar'][0] - 1, state['avatar'][1])
        if action == 'right':
            state['avatar'] = (state['avatar'][0] + 1, state['avatar'][1])
        (x, y) = state['avatar']
        in_bounds_x = self.bounds[0][0] <= x <= self.bounds[0][1]
        in_bounds_y = self.bounds[1][0] <= y <= self.bounds[1][1]
        if not in_bounds_x or not in_bounds_y:
            state['avatar'] = self.state['avatar']
        (x, y) = state['avatar']
        self.state = state
        if x > y:
            self.lost = True
        else:
            self.won = self.check_win(self.state)
        self.state['won'] = self.won
        self.state['lost'] = self.lost

    def get_obs(self):
        """
        Convert state into set of observations that agent gets to see
        """
        return deepcopy(self.state)
