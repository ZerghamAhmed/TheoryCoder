import gymnasium as gym
import minihack  # noqa: F401
import numpy as np
import os
from collections import defaultdict
import re
from copy import deepcopy
from pathlib import Path
from PIL import Image
from typing import Optional

from nle import nethack   # Needed for Command.PICKUP/APPLY/ZAP/QUAFF

# -------------------- LEVEL MAPPING --------------------

MINIHACK_LEVEL_MAPPING = {
    0: "MiniHack-Room-5x5-v0",
    1: "MiniHack-Room-15x15-v0",
    2: "MiniHack-KeyRoom-Fixed-S5-v0",
    3: "MiniHack-WoD-Medium-Full",
    4: "MiniHack-LavaCross-Levitate-Potion-Inv-Full",   # POTION LEVEL
    5: "MiniHack-LavaCross-Levitate-Ring-Pickup-Full",
    6: "MiniHack-LavaCross-Full",
    7: "MiniHack-Quest-Easy-v0",
    8: "MiniHack-Room-Trap-15x15-v0",
    9: "MiniHack-Room-Monster-15x15-v0",
}

# -------------------- utils --------------------

def _cstr(a: np.ndarray) -> str:
    b = a.tobytes()
    return b.split(b"\x00", 1)[0].decode("utf-8", "ignore").strip()

def _keyify(label: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return s or "unknown"

def _objects_from_screen_descriptions(sd: np.ndarray, *, include_floor=False):
    H, W, _ = sd.shape
    out = defaultdict(list)
    agent_xy = None
    for r in range(H):
        for c in range(W):
            label = _cstr(sd[r,c])
            if not label:
                continue
            if not include_floor and label == "floor of a room":
                continue
            xy = [c,(H-1-r)]
            k = _keyify(label)
            out[k].append(xy)
    return {k:sorted(v) for k,v in out.items()}, agent_xy


# -------------------- movement deltas --------------------

MOVE = {
    "up":(0,1),"right":(1,0),"down":(0,-1),"left":(-1,0),
    "up_right":(1,1),"down_right":(1,-1),"down_left":(-1,-1),"up_left":(-1,1),
}

# -------------------------------------------------------
#                       WRAPPER
# -------------------------------------------------------

class MinihackEnv:

    FIXED_STAIR_UP = [[36,11]]

    def __init__(self, level_set="minihack", level_id=0,
                 seed=0, include_floor=False):

        self.level_set=level_set
        self.level_id=level_id
        self.seed=seed
        self.include_floor=include_floor
        self.env_name = MINIHACK_LEVEL_MAPPING.get(level_id,"MiniHack-Room-5x5-v0")

        self.env = gym.make(
            self.env_name,
            observation_keys=("screen_descriptions","message","inv_strs","inv_letters")
        )

        # ---------------------------------------------------
        # ACTION SET HANDLING
        # ---------------------------------------------------

        # === LEVELS 3 / 7 / 5 / 6 === (unchanged)
        if level_id in (3,7,5,6):
            acts = list(self.env.unwrapped.actions)

            pickup_index = acts.index(int(nethack.Command.PICKUP))
            apply_index  = acts.index(int(nethack.Command.APPLY))
            zap_index    = acts.index(int(nethack.Command.ZAP))

            select_f_index = acts.index(ord("f"))

            north_index = acts.index(int(nethack.CompassDirection.N))
            east_index  = acts.index(int(nethack.CompassDirection.E))
            south_index = acts.index(int(nethack.CompassDirection.S))
            west_index  = acts.index(int(nethack.CompassDirection.W))

            self._action_to_id = {
                "up":0,"right":1,"down":2,"left":3,
                "up_right":4,"down_right":5,"down_left":6,"up_left":7,

                "pickup":pickup_index,
                "apply":apply_index,
                "zap":zap_index,
                "select_f":select_f_index,
                "shoot_up":north_index,
                "shoot_right":east_index,
                "shoot_down":south_index,
                "shoot_left":west_index,
            }

            self.actions_set=list(self._action_to_id.keys())
            self.action_space=gym.spaces.Discrete(len(self.actions_set))

        # === LEVEL 4 — POTION LEVEL (add quaff + select_f + potion preservation) ===
        elif level_id == 4:
            acts = list(self.env.unwrapped.actions)

            quaff_index = acts.index(int(nethack.Command.QUAFF))
            pickup_index = acts.index(int(nethack.Command.PICKUP))

            # add select_f because quaff will ask "[f or ?*]"
            select_f_index = acts.index(ord("f"))

            self._action_to_id = {
                "up":0,"right":1,"down":2,"left":3,
                "up_right":4,"down_right":5,"down_left":6,"up_left":7,

                "pickup":pickup_index,
                "quaff":quaff_index,
                "select_f":select_f_index,   # REQUIRED FOR POTION USE
            }

            self.actions_set=list(self._action_to_id.keys())
            self.action_space=gym.spaces.Discrete(len(self.actions_set))

        # === DEFAULT LEVELS ===
        else:
            self.actions_set=[
                "up","right","down","left",
                "up_right","down_right","down_left","up_left"
            ]
            self._action_to_id={a:i for i,a in enumerate(self.actions_set)}
            self.action_space=gym.spaces.Discrete(len(self.actions_set))

        # ---- STAIRS ----
        if self.env_name in ("MiniHack-Room-5x5-v0","MiniHack-Room-15x15-v0"):
            self.FIXED_STAIR_UP=[[36,11]]
        else:
            self.FIXED_STAIR_UP=[]

        self.turn_number=0
        self.won=False
        self.lost=False
        self._last_valid_flat_state=None
        self._last_agent_xy=None

        self.reset()

    # ----------------------------------------------------
    def reset(self):
        obs, info = self.env.reset(seed=self.seed)

        inv=[]
        if "inv_strs" in obs:
            inv=[_cstr(x) for x in obs["inv_strs"] if _cstr(x)]

        parsed=self._obs_to_state(obs)
        flat=self._flatten_state(parsed)

        # --- add potion inventory detection ONLY for level 4 ---
        if self.level_id == 4:
            flat["inventory"] = ["potion"] if any("potion" in x.lower() for x in inv) else []

        else:
            flat["inventory"] = ["wand"] if any("wand" in x.lower() for x in inv) else []

        self._last_valid_flat_state=deepcopy(flat)

        if "agent" in flat and flat["agent"]:
            self._last_agent_xy=flat["agent"][0]

        self.state=flat
        return deepcopy(flat)

    # ----------------------------------------------------
    def step(self, action):

        if isinstance(action,str):
            a=self._action_to_id[action.lower()]
        else:
            a=int(action)

        obs, reward, terminated, truncated, info = self.env.step(a)
        done = terminated or truncated

        inv_items=[]
        if "inv_strs" in obs:
            inv_items=[_cstr(x) for x in obs["inv_strs"] if _cstr(x)]

        # Level 4: potion inventory detection
        if self.level_id == 4:
            inv_items = ["potion"] if any("potion" in x.lower() for x in inv_items) else []
        else:
            inv_items = ["wand"] if any("wand" in x.lower() for x in inv_items) else []

        parsed=self._obs_to_state(obs)
        minotaur_dead=("a_minotaur_corpse" in parsed["objects"])
        # WoD-specific win condition: configurable via env var MINIHACK_WOD_WIN.
        #   "minotaur"  -> require minotaur kill (default — original WoD behavior)
        #   "staircase" -> require staircase reached (test if the env rewards it)
        #   "both"      -> either signal counts as a win (most permissive)
        wod_win_mode = os.environ.get("MINIHACK_WOD_WIN", "minotaur").lower()
        if self.level_id == 3:
            staircase_reached = bool(terminated and reward > 0)
            if wod_win_mode == "minotaur":
                self.won = bool(minotaur_dead)
            elif wod_win_mode == "staircase":
                self.won = staircase_reached
            else:  # "both" (default)
                self.won = bool(minotaur_dead) or staircase_reached
        else:
            # Room tasks (no minotaur present): canonical staircase signal.
            won_room = (terminated and reward > 0) if self.level_id in (0, 1, 8, 9) else False
            self.won = bool(minotaur_dead) or won_room
        self.lost=False if self.won else (done and not self.won)

        if not done:
            parsed=self._obs_to_state(obs)
            flat=self._flatten_state(parsed)
            flat["inventory"]=inv_items

            self._last_valid_flat_state=deepcopy(flat)

            if "agent" in flat and flat["agent"]:
                self._last_agent_xy=flat["agent"][0]

            self.state=flat
            return deepcopy(flat), float(reward), False, {
                "message":_cstr(obs.get("message","")),
                "inventory":inv_items
            }

        # done=True
        final=deepcopy(self._last_valid_flat_state)
        final["inventory"]=inv_items
        final["won"]=self.won
        final["lost"]=self.lost

        self.state=final
        return deepcopy(final), float(reward), True, {
            "message":_cstr(obs.get("message","")),
            "inventory":inv_items
        }

    # ----------------------------------------------------
    def _obs_to_state(self, obs):
        sd=obs["screen_descriptions"]
        objects, agent_xy = _objects_from_screen_descriptions(sd, include_floor=self.include_floor)

        if "human_caveman_called_agent" in objects:
            objects["agent"]=objects.pop("human_caveman_called_agent")

        # ---- WoD: normalize any wand variant (a_silver_wand, a_runed_wand, ...) to 'wand' ----
        # Matches the dec1_WOD_17 env behavior that the winning WM prompts assume.
        if self.level_id == 3:
            wand_coords = []
            for k in list(objects.keys()):
                if "wand" in k:
                    wand_coords.extend(objects.pop(k))
            if wand_coords:
                objects["wand"] = sorted(wand_coords)

        # ---- PRESERVE POTION IN LEVEL 4 WHEN AGENT OVERLAPS ----
        if self.level_id == 4 and self._last_valid_flat_state is not None:
            for obj_name,prev_coords in self._last_valid_flat_state.items():
                if obj_name == "agent": continue
                if obj_name in ["won","lost","inventory","staircase_up"]: continue
                if obj_name not in objects:
                    objects[obj_name] = deepcopy(prev_coords)

        inv=[]
        if "inv_strs" in obs:
            inv=[_cstr(x) for x in obs["inv_strs"] if _cstr(x)]

        return {
            "objects":objects,
            "agent":agent_xy,
            "inventory":inv,
            "won":self.won,
            "lost":self.lost,
        }

    # ----------------------------------------------------
    def _flatten_state(self, state):
        flat={}
        for obj_name,coords in state["objects"].items():
            flat[obj_name]=coords

        if self.FIXED_STAIR_UP:
            flat["staircase_up"]=deepcopy(self.FIXED_STAIR_UP)

        # Level 4: potion inventory
        if self.level_id == 4:
            flat["inventory"] = ["potion"] if any("potion" in x.lower() for x in state["inventory"]) else []
        else:
            flat["inventory"] = ["wand"] if any("wand" in x.lower() for x in state["inventory"]) else []

        flat["won"]=state["won"]
        flat["lost"]=state["lost"]
        return flat

    # ----------------------------------------------------
    def get_obs(self):
        return deepcopy(self.state)

    def close(self):
        self.env.close()

    def save_screen(self, path=None):
        img=Image.new("L",(240,40),color=255)
        Path(path or f"minihack_{self.turn_number}.png").write_bytes(img.tobytes())
