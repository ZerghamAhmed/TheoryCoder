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
        self.actions_set = ['up', 'down', 'left', 'right', 'space']
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
            stdout = subprocess.check_output(
                ['node', self.js_engine_path, str(load_pth), str(save_pth), level_set, str(level_id), move],
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
        # rows, cols = np.array(state['orig_map']).shape
        state = state["state"]["next_state"]
        # breakpoint()
        rows, cols = 10, 10
        obs = {}
        # Full map info is spread across 'obj_map' and 'back_map'
        for i in range(rows):
            for j in range(cols):
                x = state['obj_map'][j][i]
                # 'back' for 'background' maybe? Not sure what they meant,
                # but it contains e.g. the flag_obj when not found in obj_map
                xb = state['back_map'][j][i]
                # breakpoint()
                if type(x) is dict:
                    obj = x['name']
                    if x['type'] == 'phys':
                        obj += '_obj'
                    elif x['type'] == 'word':
                        obj += '_word'
                    coords = (x['x'], cols - 1 - x['y'])
                elif type(xb) is dict:
                    obj = xb['name']
                    if xb['type'] == 'phys':
                        obj += '_obj'
                    elif xb['type'] == 'word':
                        obj += '_word'
                    coords = (xb['x'], cols - 1 - xb['y'])
                else:
                    # breakpoint()
                    if xb == '_':
                        obj = 'border'
                    elif xb == ' ':
                        obj = 'empty'
                    else:
                        raise Exception(
                            'Encountered unrecognized symbol in back_map'
                        )
                    coords = (i, cols - 1 - j)
                if obj in obs.keys() and coords not in obs[obj]:
                    obs[obj].append(coords)
                else:
                    obs[obj] = [coords]

        # Check for overlaps that might have been missed
        for overlap in state['overlaps']:
            obj = overlap['name']
            if overlap['type'] == 'phys':
                obj += '_obj'
            elif overlap['type'] == 'word':
                obj += '_word'
            coords = (overlap['x'], cols - 1 - overlap['y'])

            # If the object is not in obs, or if the coords are not listed under it, add them
            if obj not in obs or coords not in obs[obj]:
                if obj in obs:
                    obs[obj].append(coords)
                else:
                    obs[obj] = [coords]

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
        # breakpoint()
        # breakpoint()
        # if len()??
        self.state = self.get_obj_coords(engine_out)  # Returns {'state': ..., 'won': ...} after initialization
        # if len(self.state['flag_word']) == 2:
            # breakpoint()
        # if self.turn_number == 11:
        #     breakpoint()
        # self.lost = not len(engine_out['state']['players'])
        # self.won = engine_out['won']
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

