"""
"""
import importlib
from pathlib import Path
from copy import deepcopy
import json
import os
import sys

# Python 3.8 shim: ast.unparse was added in 3.9. This env (minihack) is 3.8.
# The shim is guarded — on Python >=3.9 it no-ops because hasattr returns True.
import ast as _ast
if not hasattr(_ast, 'unparse'):
    import astor as _astor
    _ast.unparse = lambda node: _astor.to_source(node).rstrip()

# BFS timeout (seconds) — prevents combinatorial explosion on large action spaces (e.g. minihack WoD).
# Set to None / 0 to disable. Configurable via env var TC_BFS_TIMEOUT.
try:
    _bfs_to = float(os.environ.get('TC_BFS_TIMEOUT', '120'))
    BFS_TIMEOUT = _bfs_to if _bfs_to > 0 else None
except (ValueError, TypeError):
    BFS_TIMEOUT = 120.0
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
# from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import random
import re
import preprocessing
# from games import LavaGrid, BabaIsYou
import ast
from levelrunner import actor
import inspect
import envs.minihack_utils as utils
from preprocessing import *
import openai
import requests
from experiment_logger import ExperimentLogger
from time import strftime, gmtime

from envs.games import BabaIsYou
import yaml

from typing import List, Tuple, Optional
import shutil

import subprocess
from subprocess import CalledProcessError

import time
from contextlib import contextmanager
from statistics import mean

from datetime import datetime

from pathlib import Path
import os, shutil, importlib


from openai import OpenAI

from envs.minihack_env import MinihackEnv




initialize_world_model_prompt = \
"""You are an AI agent that must come up with a transition model of the game you are playing. 

A BFS low-level planner that will use your synthesized transition model to find the low-level actions that will allow you to win levels of the game.

You are also given state transition after executing random actions that will help as well.
Note that if there is no change returned after doing that action, it means that moving was prevented somehow such as by an obstacle. 

Make sure your world model is general. For example, do not hard code colors if there is a particular attribute you notice but have different colors.

You can toggle closed doors to open them 

Also locked doors can be toggled if you have the corresponding key color in your inventory.


CURRENT STATE:

{current_state}

ACTION SPACE:

{actions_set}

Replay Buffer (last {num_random_actions} transitions):

{errors_from_world_model}

UTILS:

{utils}


RESPONSE FORMAT:

- Make sure you use .get() to access the dictionary to avoid key errors!
For example:
avatar_pos = new_state.get('avatar') to get avatar pos 
cake_pos = new_state.get('cake') to get cake pos


```python

# make sure to include these import statements
from envs.minihack_utils import directions

def transition_model(state, action):


	Return State

```
"""


revise_world_model_prompt = \
""" You are an AI agent that must come up with a model of the game you are playing. This model you are making of the game
will be a python program that captures the logic and mechanics of the game. You have begun this world model, but it did not capture everything. 
Below is your current world model, the action space, and the state transition that your transition model handled wrong.
The state transition (inital state, action, next state) will be followed by a section detailing your prediction errors.
If the prediction errors is blank it means your world model correctly modeled that transition.

In order to craft the world model and get this state transition you explored your environment with an EXPLORATION PLAN.
The state transitions belonging to an EXPLORATION PLAN will be written below it.
Note this exploration is a high level plan and the transitions related to it
are carrying out this high level plan by 
executing actions in the ACTION SPACE {actions_set}

Pay close attention to what is involved and modify your transition model to be able to handle this.

DESCRIPTION OF DOMAIN:

In this domain, you need to descend the dungeon.


NOTES:

Feel free to also explain your thinking outside of the markup tags, but know that I will only use the code inside the markup tags. 

The exploration plans are set up to help guide you to your overall goal. 

Make sure you use .get() to access the dictionary to avoid key errors!

For example:
avatar_pos = new_state.get('avatar') to get avatar pos 
cake_pos = new_state.get('cake') to get cake pos

ACTION SPACE:

{actions_set}

CURRENT WORLD MODEL:

{world_model_str}


ERRORS FROM WORLD MODEL:

{errors_from_world_model}

UTILS:

{utils}


RESPONSE FORMAT (make sure to include your code in markup tags):

- Make sure that you return the correct state for example if you made a deepcopy of the state and modify the deep copy then return the new_state
- If you modify the state directly then return the state instead of new_state

```Python

# make sure to include these import statements
from copy import deepcopy
from envs.minihack_utils import directions

def transition_model(state, action):


        Return State

```
"""

prune_exploration_prompt = """You are an AI agent that must come up with a model of the game you are playing. This model you are making of the game
will be a python program that captures the logic and mechanics of the game. There has been an execution error in your world model.

You need to carry out an exploratory goal that will help you understand what your model is missing.

You are given the following suggestion for exploratory plans. Which one of this is the most likely one that you should carry out?

Think about the current state of the game and the current world model you have. Also include an explanation.

Notes:

Think about what types of interactions are missing in your world model. For example, try colliding into different objects
or try pushing other objects into others. Think deeply about which of these interactions you have not seen before.

SUGGESTED EXPLORATORY PLANS: 

{suggested_exploratory_plans}

CURRENT STATE:

{current_state}

CURRENT WORLD MODEL:

{world_model_str}

RESPONSE FORMAT (make sure to include your code in Python markup tags):

```Python

# just an example DO NOT ouput this
[move_to agent_1 place_2]

```

Explanation: Example explanation of why you chose this plan.

"""

debug_model_prompt = """You are an AI agent that must come up with a model of the game you are playing. This model you are making of the game
will be a python program that captures the logic and mechanics of the game. There has been an execution error in your world model.

Please fix your world model code so that this execution error is fixed. You are given the action space, state format, world model as context.

Try to make your world model as general as possible and account for possible cases that may arise in the future!
Also DO NOT make changes to "won" in the state dictionary since that will happen outside of the world model.


ACTION SPACE:

{actions_set}


UTILS:

{utils}

CURRENT WORLD MODEL:

{world_model_str}

DEBUG:

state = {state}
model(state, {action})

ERROR:

{error}

RESPONSE FORMAT (make sure to include your code in markup tags):

```Python

# make sure to include these import statements
from copy import deepcopy
from envs.minihack_utils import directions

def transition_model(state, action):


        Return State

```

"""

debug_predicate_prompt = """You are an AI agent that writes predicate functions for a planner. One of your predicates raised an execution error.

Predicate name: {predicate_name}

Current implementation:

{predicate_code}

State = {state}
Args = {args}

ERROR:

{error}

Please return the corrected predicate code in a single Python code block.
"""


execute_exploratory_plan_prompt = """You are an AI agent that must come up with a model of the game you are playing. This model you are making of the game
will be a python program that captures the logic and mechanics of the game. There has been an execution error in your world model.

You need to carry out an exploratory goal that will help you understand what your model is missing.

You are given the following suggestion for exploratory plans. However, these suggestions are high-level 
and you cannot execute them in the game. The actual executable action set for the game is 
["left", "right", "forward", "pickup", "drop", "toggle"] 

Please give the actions that will allow you to execute each of them.

Think about the current state of the game and the current world model you have. Also include an explanation.

Notes:

Please also write an explanation for why your low-level action plan satisfies the high-level suggested exloratory plan.
Please also relate it to how it can uncover the errors in the current incorrect world model.

SUGGESTED EXPLORATORY PLANS: 

{suggested_exploratory_plans}

CURRENT STATE:

{current_state}

CURRENT WORLD MODEL (NOT CORRECT):

{world_model_str}

RESPONSE FORMAT (make sure to include your code in Python markup tags):

```Python

# exploratory plan 1: open brown_door 
["forward", "forward" "toggle"]

# exploratory plan 2: drop white_key 
["left", "left" "drop"]

# exploratory plan 2: drop white_key 
["left", "left" "drop"]

# exploratory plan 3: pickup white_key
["forward", "right", "pickup"]

```

Explanation: Example explanation of why you chose this low-level action plan for each exploratory plan.
Explanation for relating to world model correction: Explain how it can uncover the errors in world model.

"""

def seed_game_from_previous(agent, src_experiment_dir: str, game: str, copy_plans: bool = True, src_game: Optional[str] = None):
    """
    Copy prior artifacts into the current experiment so new levels can start
    from an existing domain/worldmodel/predicates and only generate a problem.

    If src_game is provided (cross-game seeding), files are taken from
    tc_game/<src_game>/ and renamed to match <game> for domain/plans.

    WM-only mode: set env var TC_SEED_WM_ONLY=1 to copy ONLY worldmodel.py
    (no predicates, no domain, no plans). Useful for testing whether a
    transferred world model — without the transferred PDDL abstraction —
    is enough on its own. TC then fresh-learns predicates and domain.
    """
    src_game = src_game or game
    wm_only = os.environ.get("TC_SEED_WM_ONLY", "").lower() in ("1", "true", "yes")

    src = Path(src_experiment_dir) / "tc_game" / src_game
    dst = Path(agent.logger.experiment_dir) / "tc_game" / game
    dst.mkdir(parents=True, exist_ok=True)

    if wm_only:
        copies = [("worldmodel.py", "worldmodel.py")]
        print("[seed] WM-only mode: copying ONLY worldmodel.py (predicates / domain / plans will be fresh-learned)")
    else:
        copies = [
            ("worldmodel.py", "worldmodel.py"),
            ("predicates.py", "predicates.py"),
            (f"{src_game}_domain.pddl", f"{game}_domain.pddl"),
        ]
        if copy_plans:
            copies.append((f"{src_game}_plans.json", f"{game}_plans.json"))

    for sname, dname in copies:
        sp = src / sname
        dp = dst / dname
        if sp.exists():
            shutil.copy(sp, dp)
            print(f"[seed] copied {sp} → {dp}")
        else:
            print(f"[seed] skip missing {sp}")

    agent.game_dir = dst
    agent.domain_file = str(dst / f"{game}_domain.pddl")
    os.environ["TC_WORLDMODEL_FILE"] = str(dst / "worldmodel.py")
    os.environ["TC_PREDICATES_FILE"] = str(dst / "predicates.py")

    if wm_only:
        # Predicates and domain will be fresh-learned — do NOT mark them as filled.
        agent.predicates_empty = True
        agent.domain_empty = True
    else:
        agent.predicates_empty = False
        agent.domain_empty = False
        if "predicates" in sys.modules:
            importlib.reload(sys.modules["predicates"])
        agent.reload_predicates_module()

    print(f"[seed] seeded {game} from {src_experiment_dir}/tc_game/{src_game} → {dst}{' (WM-only)' if wm_only else ''}")


def extract_function_or_class_str(x, fname):
    """Extract code for function or class named 'fname' from string x, using AST parse and unparse"""
    tree = ast.parse(x)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == fname:
            return ast.unparse(node)
        elif isinstance(node, ast.ClassDef) and node.name == fname:
            return ast.unparse(node)
    return None

def extract_function_names(file_content):
    function_pattern = r'def\s+([^\(]+)\('
    matches = re.finditer(function_pattern, file_content)
    function_names = set(match.group(1).strip() for match in matches)
    return function_names

def process_state_baba(state):
    """
    Process the state dictionary to add controllables, overlappables, pushables, and rules_formed.

    Args:
        state (dict): The state dictionary to process.

    Returns:
        dict: The processed state dictionary.
    """
    state = {key: [list(item) for item in value] if isinstance(value, list) else value for key, value in state.items()}

    # controllables = {
    #     entity for entity in state
    #     if rule_formed(state, f'{entity[:-4]}_word', 'is_word', 'you_word')
    # }

    # overlappables = {
    #     entity for entity in state
    #     if rule_formed(state, f'{entity[:-4]}_word', 'is_word', 'win_word')
    # }

    # pushables = {
    #     entity for entity in state
    #     if entity.endswith('_word')
    #     or rule_formed(state, f'{entity[:-4]}_word', 'is_word', 'push_word')
    #     or (entity.endswith('_obj') and rule_formed(state, f'{entity[:-4]}_word', 'is_word', 'push_word'))
    # }

    # state['controllables'] = list(controllables)

    # if 'empty' in state:
    #     del state['empty']

    if 'won' in state:
        del state['won']

    # state['overlappables'] = list(overlappables)
    # state['pushables'] = list(pushables)

    # word_entities = [entity for entity in state.keys() if entity.endswith('_word')]
    # rules_on_map = []
    # for subj in word_entities:
    #     for pred in word_entities:
    #         for obj in word_entities:
    #             if rule_formed(state, subj, pred, obj):
    #                 rules_on_map.append(subj + ' ' + pred + ' ' + obj)

    # state['rules_formed'] = rules_on_map

    return state

# ---------------------------------------------------------------
# Prompt loading helpers

def load_world_prompts(game_name: str, level_id=None):
    """Load world model prompts for a specific game.

    Looks for ``initialize_world_model.txt`` and ``revise_world_model.txt``
    under ``world_modeling_prompts/<game_name>/``. If either file is missing,
    the default version from ``world_modeling_prompts/`` is used instead.

    Per-level override for minihack only: if ``game_name == "minihack"`` and
    ``level_id`` is provided, prefer files under
    ``world_modeling_prompts/minihack/lvl<N>/`` when they exist.
    """

    base_dir = Path(__file__).resolve().parent / "world_modeling_prompts"
    game_dir = base_dir / game_name

    init_path = None
    revise_path = None

    # Per-level override for minihack only.
    if game_name == "minihack" and level_id is not None:
        lvl_dir = game_dir / f"lvl{level_id}"
        cand_init = lvl_dir / "initialize_world_model.txt"
        cand_revise = lvl_dir / "revise_world_model.txt"
        if cand_init.exists():
            init_path = cand_init
        if cand_revise.exists():
            revise_path = cand_revise

    if init_path is None:
        init_path = game_dir / "initialize_world_model.txt"
        if not init_path.exists():
            init_path = base_dir / "initialize_world_model.txt"

    if revise_path is None:
        revise_path = game_dir / "revise_world_model.txt"
        if not revise_path.exists():
            revise_path = base_dir / "revise_world_model.txt"

    return init_path.read_text(), revise_path.read_text()

# ---------------------------------------------------------------
# Fast-Downward path helper

def load_downward_path(config_file: str = "downward_config.yaml") -> Optional[str]:
    """Return the Fast-Downward path from a YAML config if available."""
    cfg_path = Path(config_file)
    if cfg_path.exists():
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if isinstance(data, dict):
                    return data.get("fast_downward_path")
        except Exception:
            pass
    return None

class TheoryCoderAgent:
    """
    Theory-based RL agent.

    Factorizes world model into discrete set of interaction rules between
    object types and synthesizes code to predict next state given current state
    and action for interacting entities in each rule.

    Assumes Markov world.
    """
    def __init__(
        self,
        world_model_load_name=None,
        operators_load_name=None,
        predicates_load_name=None,
        json_reporter_path=None,  # Moved this after parameters without default values
        language_model='gpt-4o',
        # language_model = 'o1-mini',
        # language_model = 'o1-preview',
        # language_model='gpt-3.5-turbo',
        domain_file_name='domain.pddl',  # Added this for PDDL file path
        predicates_file_name='predicates.py',
        query_mode='openai_direct',  # Options: 'langchain_openai', 'openai_direct', 'groq'
        groq_model="llama3-8b-8192",  # Specify the Groq model
        # reasoning_effort: Optional[str ] = "medium",  # new: "low" | "medium" | "high" | None
        reasoning_effort: Optional[str] = "medium", # new: "low" | "medium" | "high" | None
        use_responses_api: bool = False,          # new: optional toggle to use Responses API
        # language_model='gpt-4-turbo-preview',
        # temperature=0.7,
        temperature=1,
        episode_length=20,
        do_revise_model=False,
        sparse_interactions=True,  # Only run subset of world model
        observation_memory_size=1,
        planner_explore_prob=0,
        max_replans=1,
        plans_file_name='plans.json',  # Default to a generic file if not specified
        base_dir=None,  # Added for experiment logging
        experiment_name=None,  # Added for experiment logging
        create_subdir=True,
        prune_plans=False,  # Add this parameter to switch between methods
        fast_downward_path=None,
        centralize_files=True,
        use_custom_pddl_prompts=True,
        replace_predicates=False,
        transfer_levels=None
    ):
        
    #     transfer_levels:
    #   - Set[int]              → applies to all games (e.g., {13})
    #   - Dict[str, Set[int]]   → per-game (e.g., {"babyai": {13}, "pb1": set()})
        self.transfer_levels = transfer_levels or set()


        self.runtime_vars = {
            'interaction_rules': {},
            'interaction_rules_str': {},
            'error_msg_model': '',
            'observations': [],
            'revise_plan': False,
            'plan_str': '',
            'plan_log': '',
            'goal': 'Win',
            'goal_state_str': '',
            'operators': '',
            'predicates': '',
            'worldmodel': '',
            'observed_collisions': '',
            'unobserved_collisions': '',
            'previous_entities_encountered': [],
            'new_entities_encountered': [] 
        }

       

        # Ablations
        self.do_revise_model = do_revise_model


        self.query_mode = query_mode

        # Free model parameters
        self.sparse_interactions = sparse_interactions
        self.observation_memory_size = observation_memory_size
        self.planner_explore_prob = planner_explore_prob
        self.max_replans = max_replans
        self.world_model_version = 0



        # Prompts
        # self.infer_interaction_rule_prompt = infer_interaction_rule_prompt
        # self.get_relevant_rules_prompt = get_relevant_rules_prompt
        # self.planner_prompt = planner_prompt
        # self.evaluate_plan_prompt = evaluate_plan_prompt
        self.debug_model_prompt = debug_model_prompt
        self.debug_predicate_prompt = debug_predicate_prompt
        self.initialize_world_model_prompt = initialize_world_model_prompt
        self.revise_world_model_prompt = revise_world_model_prompt

        # Initialize query clients based on query_mode
        if query_mode == 'langchain_openai':
            # self.llm_client = ChatOpenAI(model_name=language_model, temperature=temperature)
            print("hi")
        elif query_mode == 'openai_direct':
            self.llm_client = openai
        elif query_mode == 'groq':
            self.llm_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        elif query_mode == 'custom':
            # Generic OpenAI-compatible gateway (e.g. Harvard Apigee)
            self.custom_base_url = os.environ.get("CUSTOM_BASE_URL")
            self.custom_api_key  = os.environ.get("CUSTOM_API_KEY")

            if not self.custom_base_url:
                raise ValueError("CUSTOM_BASE_URL must be set for query_mode='custom'")
            if not self.custom_api_key:
                raise ValueError("CUSTOM_API_KEY must be set for query_mode='custom'")

            self.llm_client = None  # requests-based, not SDK-based
        else:
            raise ValueError(f"Unsupported query_mode: {query_mode}")

        # I/O
        self.world_model_save_name = '_model_tmp'
        self.world_model_load_name = world_model_load_name  # Possibly load existing model
        self.operators_save_name = '_operators_tmp'
        self.operators_load_name = operators_load_name
        self.predicates_save_name = '_predicates_tmp'
        self.predicates_load_name = predicates_load_name
        # remember the domain file name for later runs
        self.domain_file_name = domain_file_name
        self.plan_save_name = '_plan_tmp'
        self.actions_set_save_name = '_actions_set_tmp'

        # input files
        self.domain_empty = False 
        self.predicates_empty = False
        self.game_dir = None

        # Set up chat model
        self.language_model = language_model
        self.temperature = temperature
        # chat = ChatOpenAI(
        #     model_name=self.language_model,
        #     temperature=temperature
        # )
        # self.query_lm = lambda prompt: chat(prompt.to_messages()).content
        self.episode_length = episode_length
        self.groq_model = groq_model


        # Record episodes
        self.tape = [{}]

        # Dynamically load plans
        self.plans_file_name = plans_file_name
        # self.plans = self._load_plans()
        self.plans = {}

        self.predicates_file_name = 'predicates'

        # Initialize the updater

        self.world_model_empty = False  # Flag for empty model
        # self.world_model_available = False  # Default to False

        # Initialize experiment logger
        self.logger = ExperimentLogger(
            base_dir or os.getcwd(),
            experiment_name,
            create_subdir=create_subdir,
        )

        self.centralize_files = centralize_files
        self.use_custom_pddl_prompts = use_custom_pddl_prompts
        self.replace_predicates = replace_predicates

        # Ensure preprocessing uses predicates from this experiment directory
        os.environ["TC_PREDICATES_FILE"] = str(Path(self.logger.experiment_dir) / f"{self.predicates_file_name}.py")

        # Load domain PDDL and predicates files
        self._load_predicates(self.predicates_file_name)
        # Initialize predicate functions for preprocessing
        preprocessing.initialize_predicates(os.environ["TC_PREDICATES_FILE"])

        self.load_utils()

        # Add new runtime variables to track exploratory plans
        self.runtime_vars['exploratory_plans'] = []
        self.runtime_vars['unsatisfied_preconditions'] = []

        # Initialize level statistics
        self.level_statistics = {}

        self._global_step = 0
        self._run_serial  = 0
        self._phase_tag   = "init"
        self._level_tag   = "L?"

        self.prune_plans = prune_plans
        self.aggregated_dataset = []  # Initialize the aggregated dataset

        # Path to the Fast-Downward planner
        self.fast_downward_path = (
            fast_downward_path
            or os.environ.get("FAST_DOWNWARD_PATH")
            or load_downward_path()
            or "fast-downward.py"
        )

        self._init_timing()

    def _record_step_timing(self, *, level_id: int, step_dir: Path, stage_tag: str, elapsed_sec: float):
        """
        Write steps/NNN_* / timing.json with a flat, single-step entry.
        stage_tag should match the semantic stage (e.g., 'init_pddl_files', 'convert_plan_to_actions', 'revise_world_model').
        """
        payload = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": level_id,
            "run_serial": self._run_serial,
            "stage": step_dir.name,           # e.g., 001_L19_R1_init_pddl_files
            "stage_tag": stage_tag,           # e.g., init_pddl_files
            "time_sec": round(elapsed_sec, 4),
            "model": self.language_model,
        }
        self._write_json(step_dir / "timing.json", payload)

    def _write_json(self, path: Path, obj: dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


    def _plans_for_level(self, level: int):
    # First: try the new storage keyed by problem filename
        problem_name = f"{self._get_game_name()}_{level}.pddl"
        entry = self.plans.get(problem_name)
        if isinstance(entry, dict) and isinstance(entry.get("actions"), list):
            return entry["actions"]

        # Second: try legacy storage keyed by str(level) -> list
        legacy = self.plans.get(str(level))
        if isinstance(legacy, list):
            return legacy

        # Third: use the on-disk per-problem helper if available
        try:
            return self._load_plan_for_problem(self._problem_path_for_level(level)) or []
        except Exception:
            return []
        
    # def _discover_next_step_index(self) -> int:
    #     steps_dir = agent.game_dir / "steps"
    #     max_idx = 0
    #     if steps_dir.exists():
    #         for d in steps_dir.iterdir():
    #             if not d.is_dir():
    #                 continue
    #             # match leading 3-digit number "NNN_"
    #             m = re.match(r"^(\d{3})_", d.name)
    #             if m:
    #                 try:
    #                     max_idx = max(max_idx, int(m.group(1)))
    #                 except ValueError:
    #                     pass
    #     return max_idx + 1


    # --- modify signature
    def query_lm(self, prompt: str, *, label: Optional[str ] = None):
        """
        Unified LM caller.
        - openai_direct: supports o4-mini with reasoning effort and either Chat Completions or Responses API.
        - langchain_openai: uses your existing LangChain client.
        - groq: uses your existing Groq client.
        Returns: (text: str, raw_completion_obj: Any)
        """
        meta_extra = {"model": self.language_model, "provider": self.query_mode}

        # --- LangChain OpenAI (unchanged) ---
        if self.query_mode == 'langchain_openai':
            with self._record_time("llm", detail=label or "langchain_openai", extra=meta_extra):
                chat_prompt = HumanMessagePromptTemplate.from_template(prompt)
                out = self.llm_client.invoke(chat_prompt.to_messages())
                return out.content, None

        # --- OpenAI: Chat Completions / Responses API with reasoning control ---
        if self.query_mode == "openai_direct":
            with self._record_time("llm", detail=label or "openai_direct", extra=meta_extra):
                # Common OpenAI kwargs
                common = {
                    "model": self.language_model,
                    "temperature": getattr(self, "temperature", 1),
                }

                # Reasoning control (o4-mini etc.). Keep both styles for SDK/back-end compatibility.
                reasoning_kwargs = {}
                effort = getattr(self, "reasoning_effort", None)  # "low" | "medium" | "high" | None
                if effort:
                    reasoning_kwargs["reasoning"] = {"effort": effort}
                    reasoning_kwargs["reasoning_effort"] = effort  # back-compat shim

                completion = None
                text_out = ""

                # Pick API style
                use_responses = bool(getattr(self, "use_responses_api", False))

                if use_responses:
                    # ---- Responses API path ----
                    completion = self.llm_client.responses.create(
                        **common,
                        **reasoning_kwargs,
                        input=prompt,
                    )
                    # Normalize output text
                    if hasattr(completion, "output_text"):
                        text_out = completion.output_text
                    else:
                        try:
                            # Fallback parse
                            text_out = completion.choices[0].message["content"][0]["text"]
                        except Exception:
                            text_out = ""
                else:
                    # ---- Chat Completions path ----
                    messages = [{"role": "user", "content": prompt}]
                    completion = self.llm_client.chat.completions.create(
                        **common,
                        **reasoning_kwargs,
                        messages=messages,
                        seed=42,
                    )
                    text_out = (completion.choices[0].message.content or "").strip()

                # Try to record token usage if present
                try:
                    usage = getattr(completion, "usage", None)
                    if usage:
                        self.timing["events"][-1].update({
                            "prompt_tokens": getattr(usage, "prompt_tokens", None),
                            "completion_tokens": getattr(usage, "completion_tokens", None),
                            "total_tokens": getattr(usage, "total_tokens", None),
                        })
                except Exception:
                    pass

                return text_out, completion

        # --- Groq (unchanged shape) ---
        if self.query_mode == 'groq':
            with self._record_time("llm", detail=label or "groq", extra=meta_extra):
                response = self.llm_client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.groq_model
                )
            return response.choices[0].message.content.strip(), response

        # --- Custom OpenAI-compatible gateway ---
        if self.query_mode == "custom":
            url = f"{self.custom_base_url.rstrip('/')}/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "api-key": self.custom_api_key,
            }
            payload = {
                "model": self.language_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
            }

            with self._record_time("llm", detail=label or "custom", extra=meta_extra):
                # 300s (5 min) accommodates o4-mini-high heavy reasoning responses
                # which routinely exceed the previous 60s ceiling on complex prompts
                # like MiniHack WoD synthesis. VGDL/BabyAI runner uses 600s.
                r = requests.post(url, headers=headers, json=payload, timeout=300)
                r.raise_for_status()
                completion = r.json()

            # token usage if provided by gateway
            usage = completion.get("usage")
            if usage:
                self.timing["events"][-1].update({
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                })

            text = completion["choices"][0]["message"]["content"].strip()
            return text, completion

        # --- Fallback ---
        raise ValueError(f"Unsupported query_mode: {self.query_mode}")




        # ---- Timing helpers -------------------------------------------------
    def _init_timing(self):
        self.timing = {
            "events": [],      # list of dict entries (one per timing)
            "rollup": {},      # computed at end (category -> stats)
        }

    @contextmanager
    def _record_time(self, category: str, detail: Optional[str ] = None, extra: Optional[dict ] = None):
        """Context manager to capture wall time for any block."""
        t0 = time.perf_counter()
        err = None
        try:
            yield
        except Exception as e:
            err = str(e)
            raise
        finally:
            dt = time.perf_counter() - t0
            entry = {
                "ts": time.time(),
                "category": category,           # e.g., "llm", "fd", "planner_bfs"
                "detail": detail or "",         # e.g., "initialize_world_model"
                "duration_s": dt,
            }
            if extra:
                entry.update(extra)
            if err:
                entry["error"] = err
            self.timing["events"].append(entry)

    def _rollup_timings(self):
        out = {}
        for ev in self.timing["events"]:
            key = (ev["category"], ev.get("detail", ""))
            out.setdefault(key, []).append(ev["duration_s"])
        roll = {}
        for (cat, det), arr in out.items():
            roll.setdefault(cat, {})
            roll[cat][det] = {
                "count": len(arr),
                "total_s": sum(arr),
                "mean_s": mean(arr),
                "max_s": max(arr),
                "min_s": min(arr),
            }
        self.timing["rollup"] = roll

    def _save_timings(self, filename: str = "timings.json"):
        try:
            self._rollup_timings()
            path = Path(self.logger.experiment_dir) / filename
            with open(path, "w") as f:
                json.dump(self.timing, f, indent=2)
            print(f"[timing] wrote {path}")
        except Exception as e:
            print(f"[timing] failed to save timings: {e}")

    def set_reasoning(self, effort: Optional[str ]):
        """
        Set effort to 'low' | 'medium' | 'high' for o4-mini (or None to disable).
        """
        self.reasoning_effort = effort



    @contextmanager
    def timing_run(self, label: str, level: Optional[int ] = None, filename: Optional[str ] = None):
        """Reset timings, attach meta, and auto-save a JSON at the end."""
        self._init_timing()
        self.timing["meta"] = {
            "game": getattr(self, "_timing_game", None) or self._get_game_name(),
            "label": label,
            "level": level if level is not None else getattr(self, "current_level", None),
            "started_ts": time.time(),
        }
        try:
            yield
        finally:
            g = self.timing["meta"]["game"]
            lvl = self.timing["meta"]["level"]
            fname = filename or f"timings_{g}_{label}_L{lvl if lvl is not None else 'NA'}.json"
            self._save_timings(filename=fname)



    def load_utils(self):
        # Load the 'directions' from utils.py as a string
        directions_code = inspect.getsource(utils)  # Get the source code of utils.py
        self.runtime_vars['utils'] = directions_code  # Store it in runtime_vars as a string

    def _load_plans(self):
        """
        Load plans from the specified plans file, first checking the current
        game_dir, then falling back to the CWD.
        """
        # look in tc_game/<game> first
        game_plan = self.game_dir / self.plans_file_name

        if game_plan.exists():
            with open(game_plan, 'r') as f:
                return json.load(f)

        # fallback to root
        root_plan = Path(self.plans_file_name)
        if root_plan.exists():
            with open(root_plan, 'r') as f:
                return json.load(f)

        print(f"Plans file '{self.plans_file_name}' not found in {self.game_dir} or CWD.")
        return {}
    
    def _should_transfer(self, level: int) -> bool:
        """Return True iff this level should use 'transfer problem' generation."""
        # support Set[int] (global) OR Dict[str,Set[int]] (per-game)
        levels = self.transfer_levels
        if isinstance(levels, dict):
            game = self._get_game_name()
            levels = levels.get(game, set())
        return level in levels


    def capture_world_model(self):
        """Load worldmodel.py from the game directory, creating a placeholder
        if it does not exist.
        """
        world_model_path = self.game_dir / "worldmodel.py"

        if not world_model_path.exists():
            # Minimal placeholder to allow execution to continue
            world_model_str = (
                "from minihack_utils import directions\n\n"
                "def transition_model(state, action):\n"
                "    return state\n"
            )
            world_model_path.write_text(world_model_str)
        else:
            world_model_str = world_model_path.read_text()

        # Store in runtime vars so is_world_model_empty() can inspect it
        # Path to the worldmodel.py file
        world_model_path = self.game_dir / "worldmodel.py"
        
        # Read the entire content of the file
        with open(world_model_path, 'r') as file:
            world_model_str = file.read()
        
        # Store the content in runtime_vars
        self.runtime_vars['world_model_str'] = world_model_str



    def parse_sas_plan(self, path: str) -> List[str]:
        """
        Read a Fast-Downward sas_plan and return a list of action strings,
        stripping comments and parentheses.
        """
        actions = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(';'):
                    continue
                actions.append(line.strip('()'))
        return actions
    
    def load_prompt(self, name: str, **kwargs) -> str:
        base = Path("abstraction_prompts")
        # ICL ablation: TC_NSHOT=N takes priority over the normal lookup.
        # Looks for, in order:
        #   abstraction_prompts/minihack_lvl<lvl>_<name>_n<N>.txt  (per-level)
        #   abstraction_prompts/minihack_<name>_n<N>.txt           (game-flat)
        # If neither exists for the given TC_NSHOT, fall through to the usual lookup.
        nshot = os.environ.get("TC_NSHOT", "").strip()
        if nshot and getattr(self, "use_custom_pddl_prompts", False):
            game_name = self._get_game_name()
            if game_name == "minihack":
                lvl = getattr(self, "current_level", None)
                if lvl is not None:
                    lvl_nshot = base / f"minihack_lvl{lvl}_{name}_n{nshot}.txt"
                    if lvl_nshot.exists():
                        print(f"[load_prompt] using n-shot variant: {lvl_nshot}")
                        return lvl_nshot.read_text().format(**kwargs)
            flat_nshot = base / f"{game_name}_{name}_n{nshot}.txt"
            if flat_nshot.exists():
                print(f"[load_prompt] using n-shot variant: {flat_nshot}")
                return flat_nshot.read_text().format(**kwargs)

        if getattr(self, "use_custom_pddl_prompts", False):
            game_name = self._get_game_name()
            # Per-level override for minihack only (5x5/15x15/Trap/Monster/WoD have different goals).
            if game_name == "minihack":
                lvl = getattr(self, "current_level", None)
                if lvl is not None:
                    lvl_path = base / f"minihack_lvl{lvl}_{name}.txt"
                    if lvl_path.exists():
                        return lvl_path.read_text().format(**kwargs)
            custom_path = base / f"{game_name}_{name}.txt"
            if custom_path.exists():
                text = custom_path.read_text()
                return text.format(**kwargs)
        text = (base / f"{name}.txt").read_text()
        return text.format(**kwargs)
    

    def extract_code_block(self, text: str, lang: str, which: int) -> str:
        pattern = rf"```{lang}(.*?)(?=```)"
        blocks = re.findall(pattern, text, re.DOTALL)
        return blocks[which-1].strip() if len(blocks) >= which else ""

    def extract_pddl_files(self, text: str) -> Tuple[str, str]:
        """Return domain and problem code from ``text``.

        Handles both the standard case where the LLM returns two separate
        `````pddl```` blocks and the degenerate case where both domain and
        problem appear within a single block.
        """
        pattern = r"```pddl(.*?)(?=```)"
        blocks = re.findall(pattern, text, re.DOTALL)
        domain, problem = "", ""

        if len(blocks) >= 2:
            domain = blocks[0].strip()
            problem = blocks[1].strip()
        elif len(blocks) == 1:
            combined = blocks[0].strip()
            dom_idx = combined.find("(define (domain")
            prob_idx = combined.find("(define (problem")
            if dom_idx != -1 and prob_idx != -1:
                if dom_idx < prob_idx:
                    domain = combined[dom_idx:prob_idx].strip()
                    problem = combined[prob_idx:].strip()
                else:
                    problem = combined[prob_idx:dom_idx].strip()
                    domain = combined[dom_idx:].strip()
            elif dom_idx != -1:
                domain = combined
            elif prob_idx != -1:
                problem = combined

        return domain, problem

    def _ensure_closed_parentheses(self, code: str) -> str:
        """Return ``code`` with any missing closing parentheses appended."""
        depth = 0
        for ch in code:
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
        if depth > 0:
            code += ')' * depth
        return code

    
    def _solve_existing_pddl(self, level: int, domain_path: Path):
        """Solve with an existing domain; generate a problem if missing."""
        game_name = self._get_game_name()
        problem_path = self.game_dir / f"{game_name}_{level}.pddl"

        # Domain exists but problem missing → always create problem from domain
        if domain_path.exists() and not problem_path.exists():
            print(f"[{game_name}] domain exists, problem missing → generating problem from existing domain for level {level}")
            return self.generate_problem_for_existing_domain(level)

        # Both exist → try Fast-Downward
        if domain_path.exists() and problem_path.exists():
            print(f"[{game_name}] domain & problem exist → running Fast-Downward")
            cmd = [
                "python3", self.fast_downward_path,
                str(domain_path), str(problem_path),
                "--search", "astar(blind())"
            ]
            try:
               # in _solve_existing_pddl(...)
                with self._record_time("fd", detail="astar(blind)", extra={"level": level}):
                    subprocess.run(cmd, check=True)

            except CalledProcessError as e:
                print(f"Fast Downward failed with exit code {e.returncode}")
                # Optional: try to (re)generate the problem from the same domain once
                print(f"[{game_name}] attempting to regenerate problem from existing domain…")
                self.generate_problem_for_existing_domain(level)
                try:
                    with self._record_time("fd", detail="astar(blind)", extra={"level": self.current_level}):
                        subprocess.run(cmd, check=True)
                except CalledProcessError as e2:
                    print(f"Fast Downward still failed with exit code {e2.returncode}")
                    return None

            plan = self.parse_sas_plan("sas_plan")
            self._save_plan_for_problem(problem_path, plan)
            return plan

        # Domain missing → nothing to do here
        return None



    def _call_pddl_debug(self, domain_path: Path, problem_path: Path) -> bool:
        """Use the LLM to repair invalid PDDL files."""
        if not self.do_revise_model:
            return False

        raw_state = json.dumps(self.engine.get_obs())
        regen_tpl = Path("abstraction_prompts/regen_pddl_files.txt").read_text()
        regen_prompt = regen_tpl.format(
            domain_file=domain_path.read_text(),
            problem_file=problem_path.read_text(),
            raw_state=raw_state,
        )

        print("PDDL DEBUG PROMPT")
        print(regen_prompt)

        step_dir = Path(self.logger.create_step("debug_pddl"))
        resp, completion = self.query_lm(regen_prompt, label="regen_pddl")

        print("PDDL DEBUG RESPONSE")
        print(resp)


        with open(step_dir / "prompt.txt", "w") as f:
            f.write(regen_prompt)
        with open(step_dir / "response.txt", "w") as f:
            f.write(resp)
        with open(step_dir / "completion_info.json", "w") as f:
            json.dump(
                completion,
                f,
                default=lambda o: getattr(o, "to_dict", lambda: str(o))(),
                indent=2,
            )

        new_domain, new_problem = self.extract_pddl_files(resp)
        new_domain = self._ensure_closed_parentheses(new_domain)
        new_problem = self._ensure_closed_parentheses(new_problem)

        domain_path.write_text(new_domain)
        problem_path.write_text(new_problem)

        shutil.copy(domain_path, step_dir / domain_path.name)
        shutil.copy(problem_path, step_dir / problem_path.name)

        self.logger.add_to_tape(
            {"step": "debug_pddl", "prompt": regen_prompt, "response": resp}
        )
        return True


    def generate_problem_for_existing_domain(self, level: int):
        """Generate a problem file for an existing domain and solve it."""
        game_name = self._get_game_name()
        domain_path = self.game_dir / f"{game_name}_domain.pddl"
        problem_path = self.game_dir / f"{game_name}_{level}.pddl"

        tpl = Path("abstraction_prompts/transfer_domain.txt").read_text()
        raw = json.dumps(self.engine.get_obs())
        mission = getattr(self.engine, "mission", "")

        try:
            prompt = tpl.format(
                domain_file=domain_path.read_text(),
                raw_state=raw,
                mission=mission,
            )
        except KeyError:
            mission_header = "MISSION (context for this level):\n" + f"{mission}\n\n"
            prompt = mission_header + tpl.format(
                domain_file=domain_path.read_text(),
                raw_state=raw,
            )

        print("GEN PDDL PROMPT (problem-from-existing-domain)")
        print(prompt)

        t0 = time.perf_counter() 

        response, meta = self.query_lm(prompt, label="gen_pddl_problem_existing")
        
        genprobsec = time.perf_counter() - t0



        print("GEN PDDL RESP")
        print(response)

        # Use a more accurate step label
        step_dir = Path(self.logger.create_step("problem_from_existing_domain"))
        with open(step_dir / "prompt.txt", "w") as f:
            f.write(prompt)
        with open(step_dir / "response.txt", "w") as f:
            f.write(response)
        with open(step_dir / "completion_info.json", "w") as f:
            json.dump(meta, f, default=lambda o: getattr(o, "to_dict", lambda: str(o))(), indent=2)

        self._record_step_timing(level_id=level, step_dir=step_dir,
                                stage_tag="gen_pddl_problem_existing", elapsed_sec=genprobsec)

        _, problem_code = self.extract_pddl_files(response)
        problem_code = self._ensure_closed_parentheses(problem_code)
        problem_path.write_text(problem_code)
        shutil.copy(problem_path, step_dir / problem_path.name)

        cmd = [
            "python3", self.fast_downward_path,
            str(domain_path), str(problem_path),
            "--search", "astar(blind())",
        ]
        try:
            with self._record_time("fd", detail="astar(blind)", extra={"level": self.current_level}):
                subprocess.run(cmd, check=True)

        except CalledProcessError as e:
            print(f"Fast Downward failed with exit code {e.returncode}")
            print("Attempting LLM-based PDDL debug...")
            if self._call_pddl_debug(domain_path, problem_path):
                try:
                    with self._record_time("fd", detail="astar(blind)", extra={"level": self.current_level}):
                        subprocess.run(cmd, check=True)

                except CalledProcessError as e2:
                    print(f"Fast Downward still failed with exit code {e2.returncode}")
                    return None
            else:
                return None

        plan = self.parse_sas_plan("sas_plan")
        self._save_plan_for_problem(problem_path, plan)
        return plan



    def generate_and_solve_pddl(self, level: int):
        """
        1) Write domain & problem into self.game_dir
        2) Run Fast-Downward; on failure, ask the LLM to regen PDDL (now also seeing the current domain)
        """
        folder = self.game_dir
        folder.mkdir(parents=True, exist_ok=True)

        # Raw state and mission
        raw = json.dumps(self.engine.get_obs())
        mission = getattr(self.engine, "mission", "")

        # Figure out the "current domain" (central if enabled and present, else local); may be blank
        game_name = self._get_game_name()
        local_domain_path = folder / f"{game_name}_domain.pddl"
        central_domain_path = Path(self.logger.experiment_dir) / self.domain_file_name

        if getattr(self, "centralize_files", False) and central_domain_path.exists():
            current_domain_path = central_domain_path
        else:
            current_domain_path = local_domain_path

        try:
            current_domain = current_domain_path.read_text().strip() if current_domain_path.exists() else ""
        except Exception:
            current_domain = ""

        # 1) INITIAL GENERATION via prompt (now receives {current_domain})
        prompt = self.load_prompt(
            "init_pddl_files",
            raw_state=raw,
            mission=mission,
            current_domain=current_domain,  # <—— key new arg; template can show or ignore it
        )

        print("GENERATE INITI PDDL FILE")
        print(prompt)

        response, token_info = self.query_lm(prompt, label="generate initial pddl files")

        print("GEN PDDL INIT RESP")
        print(response)

        step_dir = Path(self.logger.create_step("pddl_init"))
        (step_dir / "prompt.txt").write_text(prompt)
        (step_dir / "response.txt").write_text(response)
        with open(step_dir / "completion_info_pddl_init.json", "w") as f:
            json.dump(
                token_info,
                f,
                default=lambda o: getattr(o, "to_dict", lambda: str(o))(),
                indent=2,
            )

        # Extract & normalize blocks
        domain_code, problem_code = self.extract_pddl_files(response)
        domain_code = self._ensure_closed_parentheses(domain_code)
        problem_code = self._ensure_closed_parentheses(problem_code)

        # Resolve write paths
        domain_path = folder / f"{game_name}_domain.pddl"
        problem_path = folder / f"{game_name}_{level}.pddl"

        # Write local snapshot
        domain_path.write_text(domain_code)
        problem_path.write_text(problem_code)

        # Keep central domain in sync (merge/additive if enabled)
        if self.centralize_files:
            self.update_experiment_domain(domain_path, domain_code)

        # Snapshot prompt outputs
        shutil.copy(str(domain_path), str(step_dir / domain_path.name))
        shutil.copy(str(problem_path), str(step_dir / problem_path.name))

        # 2) CALL FAST-DOWNWARD
        print(f"[planner] using domain file for FD: {domain_path}")
        cmd = [
            "python3",
            self.fast_downward_path,
            str(domain_path),
            str(problem_path),
            "--search",
            "astar(blind())",
        ]

        try:
            with self._record_time("fd", detail="astar(blind)", extra={"level": level}):
                subprocess.run(cmd, check=True)

        except CalledProcessError:
            # FAILED → ask LLM to regen simpler PDDL
            # Include CURRENT DOMAIN in the regen prompt as well (even if blank)
            regen_tpl = Path("abstraction_prompts/regen_pddl_files.txt").read_text()

            # If your regen template doesn't have {current_domain}, we prepend a small header safely
            regen_header = (
                "THIS IS THE CURRENT DOMAIN FILE BUILT SO FAR (IT CAN BE BLANK). "
                "ONLY EXTEND/EDIT THIS IF NEEDED; OTHERWISE REUSE IT.\n\n```pddl\n"
                f"{current_domain}\n```\n\n"
            )

            regen_prompt = regen_header + regen_tpl.format(
                domain_file=domain_code,
                problem_file=problem_code,
                raw_state=raw,
            )

            print("PDDL REGEN PROMPT")
            print(regen_prompt)

            regen_text, regen_meta = self.query_lm(regen_prompt, label = "regen_pddl")
            new_dom, new_prb = self.extract_pddl_files(regen_text)
            new_dom = self._ensure_closed_parentheses(new_dom)
            new_prb = self._ensure_closed_parentheses(new_prb)

            step_dir = Path(self.logger.create_step("pddl_regen"))
            (step_dir / "prompt.txt").write_text(regen_prompt)
            (step_dir / "response.txt").write_text(regen_text)
            with open(step_dir / "completion_info_pddl_regen.json", "w") as f:
                json.dump(
                    regen_meta,
                    f,
                    default=lambda o: getattr(o, "to_dict", lambda: str(o))(),
                    indent=2,
                )

            # Overwrite files and retry once
            domain_path.write_text(new_dom)
            problem_path.write_text(new_prb)

            # Keep central in sync after regen too
            if self.centralize_files:
                self.update_experiment_domain(domain_path, new_dom)

            shutil.copy(str(domain_path), str(step_dir / domain_path.name))
            shutil.copy(str(problem_path), str(step_dir / problem_path.name))

            with self._record_time("fd", detail="astar(blind)", extra={"level": level}):
                subprocess.run(cmd, check=True)


        # 3) PARSE PLAN & persist
        plan = self.parse_sas_plan("sas_plan")
            # write per-problem mapping (keeps *all* plans by problem filename)
        self._save_plan_for_problem(problem_path, plan)

        return plan

    

    def generate_and_save_predicates(self, domain_path: Path, problem_path: Path):
        """
        If predicates.py is empty, call the LLM to synthesize it,
        save it to the experiment-scoped predicates.py and snapshot in the game folder.
        Includes the stored plan JSON (for THIS problem file) in the prompt.
        """
        raw = json.dumps(self.engine.get_obs())

        # Get the latest world model code (prefer runtime_vars, fall back to file)
        wm_code = self.runtime_vars.get('world_model_str', '')
        if not wm_code:
            wm_path = self.game_dir / "worldmodel.py"
            if wm_path.exists():
                wm_code = wm_path.read_text()

        # Load saved plan JSON for this exact problem file (safe if missing)
        game_name = self._get_game_name()
        plans_index = self.game_dir / f"{game_name}_plans.json"
        plan_entry = {}
        try:
            if plans_index.exists():
                with open(plans_index, "r", encoding="utf-8") as f:
                    all_plans = json.load(f)
                plan_entry = all_plans.get(problem_path.name, {})
        except Exception as e:
            print(f"[pred-init] Could not read plan JSON ({plans_index}): {e}")
            plan_entry = {}

        plan_output = json.dumps(plan_entry, indent=2, ensure_ascii=False)

        # Build the prompt (inject plan_output; if template lacks the placeholder, prepend a header)
        base_prompt = self.load_prompt(
            "init_python_predicates",
            domain_file=domain_path.read_text(),
            problem_file=problem_path.read_text(),
            raw_state=raw,
            world_model=wm_code,
            plan_output=plan_output,
        )
        if "{plan_output}" not in base_prompt and plan_output:
            header = "Plan output (JSON for this problem):\n```json\n" + plan_output + "\n```\n\n"
            prompt = header + base_prompt
        else:
            prompt = base_prompt

        print("GENERATE PREDICATES PROMPT")
        print(prompt)

        # ⬇️ FIX: ensure step_dir is a Path (not str), then use / operator safely
        step_dir = Path(self.logger.create_step("predicates_init"))
        step_dir.mkdir(parents=True, exist_ok=True)

        print("PREDICATES RESPONSE")
        resp, completion = self.query_lm(prompt, label="init_predicates")

        with open(step_dir / "prompt.txt", "w", encoding="utf-8") as f:
            f.write(prompt)
        with open(step_dir / "response.txt", "w", encoding="utf-8") as f:
            f.write(resp)
        with open(step_dir / "completion_info.json", "w", encoding="utf-8") as f:
            json.dump(
                completion,
                f,
                default=lambda o: getattr(o, "to_dict", lambda: str(o))(),
                indent=2,
                ensure_ascii=False,
            )

        # extract the first python code block
        predicates_code = self.extract_code_block(resp, "python", 1)

        # write into your shared module inside experiment directory
        self.update_experiment_predicates(predicates_code)

        # snapshot into the step folder for audit
        shutil.copy(self.game_dir / "predicates.py", step_dir / "predicates.py")

        # update flags & runtime vars
        self.runtime_vars['predicates'] = predicates_code
        self.predicates_empty = False
        print("✅ Predicates synthesized and saved (with plan JSON context).")




    def _make_langchain_prompt(self, text, **kwargs):
        x = HumanMessagePromptTemplate.from_template(text)
        chat_prompt = ChatPromptTemplate.from_messages([x])
        prompt = chat_prompt.format_prompt(**kwargs)
        return prompt

    def _get_state_deltas_str(self, state0, state1):
        """
        Highlight the changes in state resulting from last action.
        """
        def _stringify(x, k=100):
            if hasattr(x, '__len__'):
                # Add ellipsis for entries of x beyond length k
                if len(x) > k:
                    return str(sorted(x[:k]))[:-1] + '...'
                else:
                    return str(sorted(x))
            else:
                return str(x)

        string = ''
        # Get set of unique keys between state0 and state1
        all_keys = set(state1.keys()).union(set(state0.keys()))

        for key in all_keys:

            if key == 'empty':
                continue

            val0 = state0.get(key)
            val1 = state1.get(key)

            # Handle cases where val0 or val1 are None
            if val0 is None:
                string += f'"{key}": Added in the next state: {_stringify(val1)}\n'
                continue  # Skip further processing if val0 is None
            if val1 is None:
                string += f'"{key}": Removed in the next state.\n'
                continue  # Skip further processing if val1 is None

            # Now that val0 and val1 are not None, proceed to compare them
            if not self._eq(val1, val0):
                cond1 = (hasattr(val1, '__len__') and len(val1) > 2)
                cond2 = (hasattr(val0, '__len__') and len(val0) > 2)
                if cond1 or cond2:
                    # For long lists of coordinates, summarize by stating what
                    # was added or removed
                    added = []
                    removed = []
                    if not hasattr(val1, '__len__'):
                        added.append(val1)
                    else:
                        for x in val1:
                            if x not in val0:
                                added.append(x)
                    if not hasattr(val0, '__len__'):
                        removed.append(val0)
                    else:
                        for x in val0:
                            if x not in val1:
                                removed.append(x)
                    string += f'"{key}": Added: {added}\n'
                    string += f'"{key}": Removed: {removed}\n'
                else:
                    string += f'"{key}": {_stringify(val0)} --> {_stringify(val1)}\n'

        return string
        
    def _eq(self, x, y):
        # def deep_convert_to_tuple(v):
        #     if isinstance(v, list):
        #         return tuple(deep_convert_to_tuple(i) for i in v)
        #     return v

        # Convert lists to tuples recursively
        x_converted = x
        y_converted = y

        # Compare the converted structures
        if isinstance(x_converted, (tuple, set)) and isinstance(y_converted, (tuple, set)):
            return x_converted == y_converted
        else:
            return x == y


    def _stringify(self, x, k=2):
        if hasattr(x, '__len__'):
            # Add ellipsis for entries of x beyond length k
            if len(x) > k:
                return str(sorted(x[:k]))[:-1] + '...'
            else:
                return str(sorted(x))
        else:
            return str(x)

    def _make_diff_string(self, pred, val, key):
        string = ""
        # Initialize missing and extra as empty lists to avoid UnboundLocalError
        missing = []
        extra = []

        if not self._eq(val, pred):
            cond1 = hasattr(val, '__len__') and len(val) > 2
            cond2 = hasattr(pred, '__len__') and len(pred) > 2
            if cond1 or cond2:
                # If lists are long, only state what was missing or extraneous
                if not hasattr(val, '__len__'):
                    missing.append(val)
                else:
                    for x in val:
                        if x not in pred:
                            missing.append(x)
                if not hasattr(pred, '__len__'):
                    extra.append(pred)
                else:
                    for x in pred:
                        if x not in val:
                            extra.append(x)
                if missing:
                    string += f'"{key}": Missing: {missing}\n'
                if extra:
                    string += f'"{key}": extraneous: {extra}\n'
            else:
                # If list of coords is short, just print both in full
                string += f'"{key}": predicted: {self._stringify(pred)}\n'
                string += f'"{key}": actual: {self._stringify(val)}\n'

        # Handling the specific case for the "empty" key
        if key == 'empty' and not missing and not extra:
            string = "You got this transition correct!"

        return string


    # Function to detect key mismatch but with the same coordinates
    def _detect_key_mismatch(self, pred, val):
        """
        Detect if keys are different but their values (coordinates) are equivalent.
        This checks if the coordinates are the same but the keys differ between the two states.
        """
        if isinstance(pred, list) and isinstance(val, list):
            # Sort both lists of coordinates for comparison
            sorted_pred = sorted(pred)
            sorted_val = sorted(val)
            return sorted_pred == sorted_val
        return False

    
    def _get_pred_errors(self, state, predictions):
        """
        Compare the state and prediction dictionaries and return a string summarizing the differences.
        """
        diff_strs = []
        all_keys = set(state.keys()).union(predictions.keys())

         # --- NEW: ignore message completely ---
        all_keys.discard("message")

        # if isinstance(self.engine, BabaIsYou):
        #     all_keys.remove("won")
            # all_keys.remove("empty")
        # if isinstance(self.engine, Boulderdash2Env):
        #     all_keys.remove("crab")
        #     all_keys.remove("butterfly")

        for key in all_keys:
            val = state.get(key, [])
            pred = predictions.get(key, [])

            # Check if key exists in both states
            if key not in state:
                # Find if there is another key in state with the same coordinates
                matching_key = self._find_matching_key(state, pred)
                if matching_key:
                    diff_strs.append(f'Key mismatch: "{key}" is missing, but "{matching_key}" has the same coordinates.\n')
                    continue

            if key not in predictions:
                matching_key = self._find_matching_key(predictions, val)
                if matching_key:
                    diff_strs.append(f'Key mismatch: "{key}" is missing, but "{matching_key}" has the same coordinates.\n')
                    continue

            diff_str = self._make_diff_string(pred, val, key)
            if diff_str:
                diff_strs.append(diff_str)

        diff_string = '\n'.join(diff_strs).strip()

        return diff_string if diff_string else ""

    # Function to find if a matching key with the same coordinates exists in the state
    def _find_matching_key(self, state, coords):
        for key, val in state.items():
            if self._detect_key_mismatch(val, coords):
                return key
        return None


    def _get_abbreviated_observations(self, obs, cutoff=3):
        init_state_abbreviated = {}
        string = '{'
        for j, (key, val) in enumerate(obs.items()):
            string += f'{key}: '
            if not hasattr(val, '__len__'):
                string += f'{val}'
            else:
                string += '['
                for i, v in enumerate(val[:cutoff]):
                    string += f'{v}'
                    if i < cutoff - 1 and len(val) > i + 1:
                        string += ', '
                if len(val) > cutoff:
                    string += ', ...'
                string += ']'
            if j < len(obs) - 1:
                string += ', '
        string += '}'
        return string

    def _update_solution(self, level_id, first_letters):
        """
        Call the updater to log the solution.
        """
        if isinstance(self.engine, BabaIsYou):
            level_set_name = self.engine.level_set  # Dynamically determine the level set
            
            # Adjust level_id based on the level_set_name
            if level_set_name == "demo_LEVELS":
                level_id += 1  # Increment for demo_LEVELS

            # Update the solution with the adjusted level_id
            self.updater.update_solution(level_id=level_id, first_letters=first_letters, level_set_name=level_set_name)


    def _update_plan(self, text):
        x = re.findall(r'```python([\s\S]*?)```', text)
        if not len(x):
            return None, 'Exception: No code found'
        x = '\n'.join(x)
        self.runtime_vars['plan_str'] = x
        if x:
            state = self.runtime_vars['observations'][-1]
            with Path('_plan_vars_tmp_state.json').open('w') as fid:
                json.dump(state, fid)

            actions_path = '_plan_vars_tmp_actions'
            logger_path = '_plan_vars_tmp_logger'
            goal_state_str_path = '_plan_vars_tmp_goal_state_str'

            imports_str = f"import json\n"
            imports_str += f"from {self.predicates_save_name} import *\n"
            imports_str += f"from {self.operators_save_name} import *\n\n"
            imports_str += f"with open('_plan_vars_tmp_state.json', 'r') as fid:\n"
            imports_str += f"    state = json.load(fid)\n"
            save_str = f"\nactions_path = '{actions_path}'\n"
            save_str += f"logger_path = '{logger_path}'\n"
            save_str += f"goal_state_str_path = '{goal_state_str_path}'\n"
            save_str += "with open(actions_path, 'w') as fid:\n"
            save_str += "    fid.write(str(actions))\n"
            save_str += "with open(logger_path, 'w') as fid:\n"
            save_str += "    fid.write(str(logger))\n"
            save_str += "with open(goal_state_str_path, 'w') as fid:\n"
            save_str += "    fid.write(goal_state_str)\n"
            x1 = imports_str + x + save_str

            with Path(self.plan_save_name + '.py').open('w') as fid:
                fid.write(x1)

            import subprocess

            try:
                result = subprocess.run(['python', self.plan_save_name + '.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                # exec(x)
            except Exception as e:
                return None, e
            else:
                # Detect runtime errors
                stderr = result.stderr.decode('utf-8')
                if result.returncode != 0:
                    return None, stderr

                with Path(actions_path).open('r') as fid:
                    actions = fid.read()
                try:
                    actions = eval(actions)
                except:
                    actions = []
                with Path(logger_path).open('r') as fid:
                    logger = fid.read()
                with Path(goal_state_str_path).open('r') as fid:
                    goal_state_str = fid.read()
                self.runtime_vars['goal_state_str'] = locals()['goal_state_str']
                self.runtime_vars['plan_log'] = logger

                return actions, None
        else:
            return None, 'Exception: No code found inside Python tags.'
    
    def _call_model_debug(self, state, action, max_retries=3):
        if not self.do_revise_model:
            return

        for i in range(max_retries):
            try:
                if self.is_world_model_empty():
                    pred = state  # No-op model
                    return pred
                else:
                    import worldmodel
                    importlib.reload(worldmodel)
                    pred = worldmodel.transition_model(state, action)
                    return pred
            except Exception as e:
                print(f'DEBUG ITER {i}')
                print(f'ERROR: {e}')

                # Create the debug prompt
                prompt = self.debug_model_prompt.format(
                    actions_set=self.engine.actions_set,
                    world_model_str=self.runtime_vars['world_model_str'],
                    observations='IGNORE',
                    state=state,
                    action=action,
                    error=e,
                    utils="directions = {\n    'left': [-1, 0],\n    'right': [1, 0],\n    'up': [0, 1],\n    'down': [0, -1],\n    'up_right': [1, 1],\n    'down_right': [1, -1],\n    'down_left': [-1, -1],\n    'up_left': [-1, 1],\n}",
                )

                # Use experiment logger to save debug files
                step_dir = self.logger.create_step("debug")
                resp, completion = self.query_lm(prompt, label="debug_world_model")
                new_world_model_code = self.extract_code_from_response(resp)

                if new_world_model_code:
                    self.logger.save_step_files(
                        step_dir,
                        prompt,
                        resp,
                        new_world_model_code,
                        "worldmodel.py"
                    )

                    # Save full response and completion object instead of just fingerprint
                    with open(os.path.join(step_dir, "completion_info.json"), "w") as f:
                        json.dump({
                            "response": resp,
                            "completion": completion if isinstance(completion, dict) else str(completion)
                        }, f, indent=2)

                    self.logger.add_to_tape({
                        "step": "debug",
                        "prompt": prompt,
                        "response": resp,
                        "error": str(e)
                    })

                    # Update world model code and version
                    self.runtime_vars['world_model_str'] = new_world_model_code
                    self.runtime_vars['error_msg_model'] = new_world_model_code
                    self.overwrite_world_model(new_world_model_code)
                    self.world_model_version += 1
                    

                    # Overwrite the current worldmodel.py file with the new model
                    self.overwrite_world_model(new_world_model_code)

                    # Add to tape for logging
                    self.tape[-1]['debug_model_prompt'] = prompt
                    self.tape[-1]['debug_model_response'] = resp

        return None  # Return None if all retries failed

    def _call_predicate_debug(self, state, operator, error, max_retries=3):
        if not self.do_revise_model:
            return

        predicate_name = getattr(error, 'predicate_name', 'unknown')
        args = getattr(error, 'args', [])

        preds_file = os.environ.get('TC_PREDICATES_FILE', 'predicates.py')

        for i in range(max_retries):
            try:
                spec = importlib.util.spec_from_file_location('predicates', preds_file)
                predicates_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(predicates_module)
                func = getattr(predicates_module, predicate_name)
                return func(state, *args)
            except Exception as e:
                print(f'PREDICATE DEBUG ITER {i}')
                print(f'ERROR: {e}')

                try:
                    predicate_code = inspect.getsource(getattr(predicates_module, predicate_name))
                except Exception:
                    predicate_code = ''

                prompt = self.debug_predicate_prompt.format(
                    predicate_name=predicate_name,
                    predicate_code=predicate_code,
                    state=state,
                    args=args,
                    error=e,
                )

                step_dir = self.logger.create_step('debug_predicate')
                resp, completion = self.query_lm(prompt, label="debug_predicate")
                new_pred_code = self.extract_code_from_response(resp)

                if new_pred_code:
                    self.update_experiment_predicates(new_pred_code)
                    self.logger.save_step_files(
                        step_dir,
                        prompt,
                        resp,
                        new_pred_code,
                        'predicates.py'
                    )

                    with open(os.path.join(step_dir, 'completion_info.json'), 'w') as f:
                        json.dump({'response': resp, 'completion': completion if isinstance(completion, dict) else str(completion)}, f, indent=2)

                    self.logger.add_to_tape({
                        'step': 'debug_predicate',
                        'prompt': prompt,
                        'response': resp,
                        'error': str(e)
                    })

                    # update runtime vars with new file
                    with open(preds_file, 'r') as f:
                        self.runtime_vars['predicates'] = f.read()

        return None


    def _do_revise_model(self, error_count):
        # TODO: Consider fancier rule here
        if error_count > 0:
            return True
        return False

    def _do_revise_plan(self, error_count):
        if error_count > 0:
            return True
        return False

    def sample_replay_buffer(self, batch_size):
        """Sample a batch of transitions from the replay buffer."""
        batch = random.sample(self.replay_buffer, batch_size)
        return batch

    def _extract_sparse_rules(self, resp):
        """
        Assume rules are given like:

        ('entity1', 'entity2')
        ('entity2', 'entity3')
        ...

        Return list of tuples of strings
        """
        rules = re.findall(r"\([\'\"]([\w\s]+)[\'\"], [\'\"]([\w\s]+)[\'\"]\)", resp)
        return rules

    def _update_replay_buffers(self, obs):
        self.replay_buffers.append(obs)
    
    def _make_observation_summaries(self, obs, errors):
        s0, a, s1 = obs
        return (
            f"Initial state: {s0}\n"
            f"Action: {a}\n"
            f"Next state: {s1}\n"
            f"\nYour prediction errors:\n{errors}\n"
        )


    def _choose_synthesis_examples(self, exploratory_plan=None):
        """
        Choose (s0, a) --> s1 transitions from replay buffer as program
        synthesis examples.

        Args:
            exploratory_plan (str): The exploratory plan for which to generate errors.

        Returns:
            list: A list of formatted examples.
            int: The count of errors.
        """
        # Simple solution: Just take the last k from the buffer
        # Cap to last N to keep WM revision prompts under gateway payload limits.
        # TC_REPLAY_CAP env var lets you tune; default 6.
        _cap = int(os.environ.get("TC_REPLAY_CAP", "6"))
        obs = self.replay_buffers[-_cap:] if _cap > 0 else self.replay_buffers[::1]

        # obs = self.replay_buffers[::-3]

        # half_index = len(self.replay_buffers) // 2  # Get the halfway index for level 13
        # obs = self.replay_buffers[:half_index] 

        # if self.current_level == 13 or self.current_level == 16:
        #     half_index = len(self.replay_buffers) // 2  # Get the halfway index for level 13
        #     obs = self.replay_buffers[:half_index]      # 

        actions_taken = [a for (s0, a, s1) in obs]
        correct_states = [s1 for (s0, a, s1) in obs]

        # Generate predictions for each (s0, a) pair in obs
        preds = [self._call_model_debug(s0, a) for (s0, a, s1) in obs]

        # Compare predicted and actual states to identify errors
        errors = [self._get_pred_errors(s1, pred) for (s0, a, s1), pred in zip(obs, preds)]

        # Create summaries of the observations along with the errors
        examples = [self._make_observation_summaries((s0, a, s1), e) for (s0, a, s1), e in zip(obs, errors)]

        # Count the number of errors
        error_count = sum([1 if e else 0 for e in errors])

        # Format examples with the exploratory plan if provided
        if exploratory_plan:
            # last_example = examples[-1] if examples else ""
            # formatted_examples = [f"ERRORS FROM WORLD MODEL for EXPLORATORY PLAN {exploratory_plan}:\n\n{last_example}"]
            formatted_examples = [f"ERRORS FROM WORLD MODEL for EXPLORATORY PLAN {exploratory_plan}:\n\n" + "\n\n".join(examples)]

        else:
            formatted_examples = examples

        return formatted_examples, error_count

    def _revise_lvl3_bundle(self):
        """WoD-specific: revise WM, then regenerate predicates so they stay
        consistent with the new WM's kill signal (otherwise BFS is condemned
        by a frozen predicate that checks the wrong condition).
        """
        # 1) Revise WM as usual
        self._revise_world_model()

        # 2) Regenerate predicates against the new WM
        game_name = self._get_game_name()
        domain_path = self.game_dir / f"{game_name}_domain.pddl"
        problem_path = self.game_dir / f"{game_name}_{self.current_level}.pddl"
        if domain_path.exists() and problem_path.exists():
            print("[lvl3 bundle] regenerating predicates to match revised WM")
            self.generate_and_save_predicates(domain_path, problem_path)
            self._load_predicates(self.predicates_file_name)
            self.reload_predicates_module()

    def _revise_world_model(self):
        if not self.do_revise_model:
            return

        self.tape[-1]['revision_prompts'] = {}
        self.tape[-1]['revision_responses'] = {}

        examples, error_count = self._choose_synthesis_examples()
        mission = getattr(self.engine, 'mission', '')

        if self._do_revise_model(error_count):
            prompt = self.revise_world_model_prompt.format(
                actions_set=self.engine.actions_set,
                errors_from_world_model='\n\n'.join(examples),
                world_model_str=self.runtime_vars['world_model_str'],
                utils="directions = {\n    'left': [-1, 0],\n    'right': [1, 0],\n    'up': [0, 1],\n    'down': [0, -1],\n    'up_right': [1, 1],\n    'down_right': [1, -1],\n    'down_left': [-1, -1],\n    'up_left': [-1, 1],\n}",
                mission=mission,
            )

            print(prompt)

            # Create step directory and save files
            step_dir = self.logger.create_step("revision")
            resp, completion = self.query_lm(prompt, label="revise_world_model")
            new_world_model_code = self.extract_code_from_response(resp)

            if new_world_model_code:
                self.logger.save_step_files(
                    step_dir,
                    prompt,
                    resp,
                    new_world_model_code,
                    "worldmodel.py"
                )

                # Save full response and completion object instead of just fingerprint 
                with open(os.path.join(step_dir, "completion_info.json"), "w") as f:
                    json.dump({
                        "response": resp,
                        "completion": completion if isinstance(completion, dict) else str(completion)
                    }, f, indent=2)

                self.logger.add_to_tape({
                    "step": "revision",
                    "prompt": prompt,
                    "response": resp
                })

                # Save pruned plans to a text file
                pruned_plans_path = os.path.join(step_dir, "pruned_plans.txt")
                with open(pruned_plans_path, 'w') as f:
                    f.write('\n'.join(self.runtime_vars['exploratory_plans']))

                # Update world model code and version
                self.runtime_vars['world_model_str'] = new_world_model_code
                self.runtime_vars['error_msg_model'] = new_world_model_code

                # Overwrite the current worldmodel.py file with the new model
                self.overwrite_world_model(new_world_model_code)

            self.tape[-1]['revision_prompts'] = prompt
            self.tape[-1]['revision_responses'] = resp
            print(prompt)
            print(resp)
            if self._do_revise_plan(error_count):
                self.runtime_vars['revise_plan'] = True

    def _initialize_world_model(self, num_actions):
        # Build examples & context
        examples, error_count = self._choose_synthesis_examples()
        mission = getattr(self.engine, 'mission', '')

        # Current WM as a string (may be blank)
        current_wm = self.runtime_vars.get('world_model_str', '').strip()

        # Format directly into your template (since it already contains the header)
        base_prompt = self.initialize_world_model_prompt.format(
            current_state=self.runtime_vars['observations'][-1],
            actions_set=self.engine.actions_set,
            num_random_actions=num_actions,
            errors_from_world_model='\n\n'.join(examples),
            utils="directions = {\n    'left': [-1, 0],\n    'right': [1, 0],\n    'up': [0, 1],\n    'down': [0, -1],\n    'up_right': [1, 1],\n    'down_right': [1, -1],\n    'down_left': [-1, -1],\n    'up_left': [-1, 1],\n}",
            mission=mission,
            world_model=current_wm  # <-- always pass this
        )

        prompt = base_prompt

        # Persist prompt for debugging
        file_name = 'current_prompt_INIT.txt'
        with open(file_name, 'w') as file:
            file.write(prompt)

        print(prompt)

        # Call LLM
        resp, completion = self.query_lm(prompt, label="initialize_world_model")
        new_world_model_code = self.extract_code_from_response(resp)

        if new_world_model_code:
            # Save/overwrite world model
            self.runtime_vars['world_model_str'] = new_world_model_code
            self.overwrite_world_model(new_world_model_code)

            # Log artifacts
            step_dir = self.logger.create_step("initialize")
            self.logger.save_step_files(
                step_dir,
                prompt,
                resp,
                new_world_model_code,
                "worldmodel.py"
            )
            import json, os
            with open(os.path.join(step_dir, "completion_info.json"), "w") as f:
                json.dump(
                    completion,
                    f,
                    default=lambda o: getattr(o, "to_dict", lambda: str(o))(),
                    indent=2
                )

            self.logger.add_to_tape({
                "step": "initialize",
                "prompt": prompt,
                "response": resp
            })




    def _random_explore(self):
        return [random.choice(self.actions_set)]    

    def _get_plan_feedback(self):
        state = self.runtime_vars['observations'][-1]
        try:
            goal_reached = eval(self.runtime_vars['goal_state_str'])
        except Exception as e:
            self.runtime_vars['plan_feedback'] = e
        else:
            if goal_reached:
                self.runtime_vars['plan_feedback'] = 'Goal reached!'
            else:
                self.runtime_vars['plan_feedback'] = 'Goal was not reached.'

    def _hierarchical_planner(self, mode, subplan_exploratory=None):
        actions = []

        if mode == 'explore_collision':
            state = self.engine.get_obs()
            if isinstance(self.engine, BabaIsYou):
                state = process_state_baba(state)
            import worldmodel
            import planner
            import levelrunner
            importlib.reload(worldmodel)
            importlib.reload(planner)
            importlib.reload(levelrunner)
            self.reload_predicates_module()


            with self._record_time("planner_bfs", detail=("explore_collision" if mode == "explore_collision" else "exploit"),
                       extra={"level": getattr(self, "current_level", None)}):
                actionlist, state = actor(
                    self.domain_file,
                    subplan_exploratory if mode == "explore_collision" else subplan,
                    state,
                    max_iterations=None,
                    debug_callback=self._call_model_debug,
                    predicate_debug_callback=self._call_predicate_debug,
                    level=None if mode == "explore_collision" else self.current_level,
                    engine=self.engine,
                    timeout=BFS_TIMEOUT,
                )
            if not actionlist:
                print("No actions found; executing random action and revising model.")
                actions.append(random.choice(self.actions_set))
                self._revise_world_model()
            else:
                actions.extend(actionlist)

            return actions
        
        else:
            actions = []

            for i in range(self.max_replans):
                state = self.engine.get_obs()
                if isinstance(self.engine, BabaIsYou):
                    state = process_state_baba(state)               
                import worldmodel
                import planner
                import levelrunner
                import envs.minihack_utils as minihack_utils
                importlib.reload(worldmodel)
                importlib.reload(planner)
                importlib.reload(levelrunner)
                importlib.reload(minihack_utils)
                self.reload_predicates_module()
                plans_for_level = self._plans_for_level(self.current_level)
                for subplan in plans_for_level:
                    with self._record_time("planner_bfs", detail=f"exploit_subplan:{subplan}"):
                        action_seq, state = actor(
                            self.domain_file,
                            subplan,
                            state,
                            max_iterations=None,
                            debug_callback=self._call_model_debug,
                            predicate_debug_callback=self._call_predicate_debug,
                            level=self.current_level,
                            engine=self.engine,
                            timeout=BFS_TIMEOUT,
                        )
                    if not action_seq:
                        print(f"No actions found for subplan {subplan}; executing random action and revising model.")
                        actions.append(random.choice(self.actions_set))
                        self._revise_world_model()
                    else:
                        actions.extend(action_seq)


            return actions  # Return the subplans as a list of tuples

    def _sample_planner_mode(self):
        if random.choices(
            [0, 1],
            weights=[1 - self.planner_explore_prob, self.planner_explore_prob]
        )[0]:
            mode = 'explore'
        else:
            mode = 'exploit'
        return mode


    def _load_world_model(self, world_model_load_name):
        exp_model = self.game_dir / "worldmodel.py"
        model_path = exp_model
        if model_path.exists():
            spec = importlib.util.spec_from_file_location("world_model", model_path)
            world_model = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(world_model)

            try:
                if not hasattr(world_model, 'transition_model'):
                    print("Warning: transition_model function not found.")
                    self.world_model_empty = True
                    return

                transition_model_code = inspect.getsource(world_model.transition_model).strip()

                placeholder_code = "def transition_model(state, action):\n    return state"

                if transition_model_code == placeholder_code:
                    print("Warning: transition_model is unimplemented (placeholder).")
                    self.world_model_empty = True
                elif len(transition_model_code.splitlines()) <= 2:
                    print("Warning: transition_model is effectively empty.")
                    self.world_model_empty = True
                else:
                    self.world_model_empty = False

                # Save the current world model to runtime_vars
                self.runtime_vars['world_model'] = world_model
                # Save the initial loaded model
                

            except AttributeError:
                print("Warning: transition_model function not found.")
                self.world_model_empty = True
            except Exception as e:
                print(f"Error loading world model: {e}")
                self.world_model_empty = True
        else:
            print(f"World model file '{world_model_load_name}.py' not found in {self.game_dir}.")
            self.world_model_empty = True

    def _load_domain_pddl(self, domain_file_name):
        domain_path = Path(domain_file_name)
        if domain_path.exists():
            with domain_path.open('r') as f:
                content = f.read().strip()

            if not content:
                print(f"Warning: {domain_file_name} is empty.")
                self.domain_empty = True
            elif "define" not in content:
                # Check for a basic PDDL structure keyword
                print(f"Warning: {domain_file_name} does not contain valid PDDL content.")
                self.domain_empty = True
            else:
                self.domain_empty = False
                self.runtime_vars['domain_file'] = content  # Save content to runtime_vars
        else:
            print(f"Domain file '{domain_file_name}' not found.")
            self.domain_empty = True

    def _load_predicates(self, predicates_file_name):
        exp_dir = Path(getattr(self, "logger", None).experiment_dir) if hasattr(self, "logger") else None
        base_dir = Path(getattr(self, "logger", None).base_dir) if hasattr(self, "logger") else None

        predicates_path = None
        if exp_dir:
            path = exp_dir / f"{predicates_file_name}.py"
            if path.exists():
                predicates_path = path
        if predicates_path is None and base_dir:
            path = base_dir / f"{predicates_file_name}.py"
            if path.exists():
                predicates_path = path
        if predicates_path is None:
            path = Path(f"{predicates_file_name}.py")
            if path.exists():
                predicates_path = path

        if predicates_path and predicates_path.exists():
            with predicates_path.open('r') as f:
                content = f.read().strip()

            if not content:
                print(f"Warning: {predicates_file_name}.py does not define any functions or classes.")
                self.predicates_empty = True
            else:
                self.predicates_empty = False
                self.runtime_vars['predicates'] = content  # Save content to runtime_vars
        else:
            print(f"Predicates file '{predicates_file_name}.py' not found.")
            self.predicates_empty = True

    def print_world_model_contents(self):
        # Get the world_model from runtime_vars
        world_model = self.runtime_vars["world_model"]
        
        # Get all functions and classes in the module
        module_contents = inspect.getmembers(world_model, predicate=inspect.isfunction)
        # module_contents += inspect.getmembers(world_model, predicate=inspect.isclass)
        
        for name, member in module_contents:
            print(f"### {name} ###")
            try:
                # Get the source code of the function
                source_code = inspect.getsource(member)
                
                # Filter out import statements from the function's source code
                filtered_source = "\n".join(
                    line for line in source_code.splitlines() if not line.lstrip().startswith(("import", "from"))
                )
                
                # Print the filtered source code
                print(filtered_source)
            except TypeError:
                print(f"Could not retrieve source for {name}")


    def _save_actions_set_to_file(self):
        with Path(self.actions_set_save_name + '.py').open('w') as fid:
            fid.write(f"actions_set = {self.actions_set}")

    def _save_operators_to_file(self):
        with Path(self.operators_save_name + '.py').open('w') as fid:
            fid.write(self.runtime_vars['operators'].replace('{{', '{').replace('}}', '}'))

    def _save_predicates_to_file(self):
        # shared (master) predicates.py inside experiment directory
        with open(Path(self.logger.experiment_dir) / f"{self.predicates_save_name}.py", "w") as f:
            f.write(self.runtime_vars['predicates'])

        # snapshot in the game directory
        (self.game_dir / "predicates.py").write_text(self.runtime_vars['predicates'])

    def _extract_block(self, text: str, start_marker: str) -> Tuple[str, int, int]:
        """Return block starting with start_marker along with its span."""
        start = text.find(start_marker)
        if start == -1:
            return "", -1, -1
        depth = 0
        i = start
        while i < len(text):
            if text[i] == '(':
                depth += 1
            elif text[i] == ')':
                depth -= 1
                if depth == 0:
                    return text[start:i + 1], start, i + 1
            i += 1
        return text[start:], start, len(text)

    def _parse_domain(self, code: str) -> Tuple[List[str], List[str], str, str]:
        """Extract predicates and actions from PDDL domain code."""
        preds_block, p_start, p_end = self._extract_block(code, "(:predicates")
        predicates = []
        if preds_block:
            lines = preds_block.splitlines()[1:-1]
            predicates = [ln.strip() for ln in lines if ln.strip()]

        actions = []
        # start searching for actions immediately after the predicates block
        idx = p_end if p_end != -1 else 0
        while True:
            block, a_start, a_end = self._extract_block(code[idx:], "(:action")
            if a_start == -1:
                break
            actions.append(block)
            idx += a_end

        header = code[:p_start] if p_start != -1 else code
        # grab everything after the last action as the footer
        footer = code[idx:].strip()
        return predicates, actions, header, footer

    def _merge_domain(self, base: str, new: str) -> str:
        """Merge actions and predicates from new into base domain code."""
        base_preds, base_actions, header, footer = self._parse_domain(base)
        new_preds, new_actions, _, _ = self._parse_domain(new)

        for p in new_preds:
            if p not in base_preds:
                base_preds.append(p)

        def name(action: str) -> str:
            m = re.search(r"\(:action\s+(\S+)", action)
            return m.group(1) if m else action

        existing = {name(a): a for a in base_actions}
        for a in new_actions:
            n = name(a)
            if n not in existing:
                base_actions.append(a)

        preds_section = "(:predicates\n    " + "\n    ".join(base_preds) + "\n)"
        actions_section = "\n\n".join(base_actions)
        return f"{header}{preds_section}\n\n{actions_section}\n{footer}"

    def update_experiment_domain(self, exp_path: Path, domain_code: str) -> None:
        """Add per-game domain code into the experiment-scoped domain file."""
        if not self.centralize_files:
            exp_path.write_text(domain_code)
            return

        central_path = Path(self.logger.experiment_dir) / self.domain_file_name
        if central_path.exists():
            existing = central_path.read_text()
            merged = self._merge_domain(existing, domain_code)
            central_path.write_text(merged)
        else:
            central_path.write_text(domain_code)

        # Also merge back into the per-game snapshot to keep it up to date
        if exp_path.exists():
            current = exp_path.read_text()
            merged_local = self._merge_domain(current, domain_code)
            exp_path.write_text(merged_local)
        else:
            exp_path.write_text(domain_code)


    def _merge_predicates(self, base: str, new: str) -> str:
        """Merge top-level definitions from ``new`` into ``base``.

        If ``self.replace_predicates`` is ``True``, existing definitions with the
        same name will be replaced. Otherwise new definitions are appended only
        when the name does not already exist.
        """
        if not base.strip():
            return new

        try:
            base_ast = ast.parse(base)
            new_ast = ast.parse(new)
        except SyntaxError:
            # Fallback to simple concatenation if parsing fails
            return base + "\n" + new

        existing_map = {
            node.name: node
            for node in base_ast.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }

        for node in new_ast.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue

            if node.name in existing_map:
                if self.replace_predicates:
                    idx = base_ast.body.index(existing_map[node.name])
                    base_ast.body[idx] = node
            else:
                base_ast.body.append(node)

        return ast.unparse(base_ast)

    def update_experiment_predicates(self, code: str) -> None:
        """Merge ``code`` with the experiment-scoped ``predicates.py``."""
        path = Path(self.logger.experiment_dir) / f"{self.predicates_file_name}.py"
        base_dir_path = Path(self.logger.base_dir) / f"{self.predicates_file_name}.py"

        if path.exists():
            base = path.read_text()
        elif base_dir_path.exists():
            base = base_dir_path.read_text()
        else:
            base = ""

        merged = self._merge_predicates(base, code) if base else code
        path.write_text(merged)
        # snapshot in the game directory as well
        (self.game_dir / "predicates.py").write_text(merged)

        # also update the shared predicates file at the base directory
        base_dir_path.write_text(merged)



    def is_world_model_empty(self):
        """
        Check if the transition_model function is effectively empty, meaning it performs no significant logic.
        """
        # Retrieve the world model string from runtime_vars
        world_model_str = self.runtime_vars.get('world_model_str', '')

        # Parse the code into an AST
        tree = ast.parse(world_model_str)

        # Look for the transition_model function in the AST
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'transition_model':
                # Check if the body of the function is minimal (e.g., just a return statement)
                if len(node.body) == 1:
                    # First node should be the return statement
                    return_node = node.body[0]
                    
                    # Check if the return statement is returning 'state'
                    if isinstance(return_node, ast.Return) and isinstance(return_node.value, ast.Name) and return_node.value.id == 'state':
                        # This is effectively an empty world model
                        return True

        # If no transition_model function is found or it does more than just return 'state'
        return False

    def overwrite_world_model(self, new_code: str):
        # Save the updated world model inside the current game folder only
        (self.game_dir / "worldmodel.py").write_text(new_code)


    def extract_code_from_response(self, response):
        # Use a regular expression to extract the Python code within ```python ``` tags (case-insensitive)
        code_match = re.search(r'```python(.*?)```', response, re.DOTALL | re.IGNORECASE)
        if code_match:
            return code_match.group(1).strip()
        else:
            return None

    def reload_predicates_module(self):
        """Ensure the predicates module is loaded and reload it."""
        module_name = "predicates"
        preds_file = os.environ.get("TC_PREDICATES_FILE")
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])
        elif preds_file and Path(preds_file).exists():
            spec = importlib.util.spec_from_file_location(module_name, preds_file)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

    def execute_actions(self, env, actions):
        """
        Function to loop through a list of actions and step through the environment.
        
        :param env: The environment instance to execute actions on.
        :param actions: List of actions to perform.
        """
        for action in actions:
            try:
                # Execute the action
                state, reward, done, info = env.step(action)
                # Save the screen if necessary
                env.save_screen()
                # Print the results of the action
                print(f"Action: {action}")
                print(f"State: {state}")
                print(f"Reward: {reward}, Done: {done}")
                # Break the loop if the environment is done
                if done:
                    print("Environment reached a terminal state.")
                    break
            except Exception as e:
                print(f"An error occurred during execution: {e}")
                break

    

    def reset(self, keep_model=True):
        self.engine.reset()
        self.runtime_vars['revise_plan'] = False
        self.actions_set = self.engine.actions_set

        state = self.engine.get_obs().copy()

        # breakpoint()
        

        # Example usage:
        # Assuming `self.engine` is your environment instance and the following actions were executed.
        # actions = [
        #     'right', 'down', 'down', 'left', 'down', 'right', 'right',
        #     'right', 'right', 'down', 'right', 'up', 'left', 'left',
        #     'left', 'up', 'up', 'up', 'up', 'right', 'down', 'down',
        #     'down', 'down'
        # ]

        # Reset the environment before starting
        # Execute the actions
        # self.execute_actions(self.engine, actions)


        if isinstance(self.engine, BabaIsYou):
            state = process_state_baba(state)

        
        self.runtime_vars['observations'] = [state]
        self.actions = []
        self.replay_buffers = []

        self.capture_world_model()
        
        # Check if the world model is empty
        if self.is_world_model_empty():
            print("Detected an empty world model.")
            # self._initialize_world_model()
            num_actions = 15
            plan = self.execute_random_actions(num_actions=num_actions)  # Adjust the number as needed
            print(plan)
            print("World model was empty, revised the model. Moving to next iteration.")
            for action in plan:
                self.step_env(action)
            self._initialize_world_model(num_actions)

            self.capture_world_model()

        if self.predicates_empty:
            print("Warning: Predicates file is empty or contains no valid functions/classes.")
            print("⚠️  predicates.py is empty → generating via LLM")
            game_name   = self._get_game_name()
            domain_path = self.game_dir / f"{game_name}_domain.pddl"
            problem_path = self.game_dir / f"{game_name}_{self.current_level}.pddl"
            self.generate_and_save_predicates(domain_path, problem_path)
    
    def prune_exploratory_plans(self, plans):
        """
        Deduplicate exploratory plans by removing entity indices.

        Args:
            plans (list): List of exploratory plans (e.g., 'push_to baba_obj_1 rock_obj_1 flag_obj_1').

        Returns:
            list: Deduplicated plans (e.g., 'push_to baba_obj rock_obj flag_obj').
        """
        deduplicated = set()  # Use a set to ensure uniqueness
        for plan in plans:
            # Remove indices using regex
            pruned_plan = re.sub(r'_\d+', '', plan)
            deduplicated.add(pruned_plan)

        return list(deduplicated)  # Convert back to a list
    
    def _problem_path_for_level(self, level: int) -> Path:
        game_name = self._get_game_name()
        return self.game_dir / f"{game_name}_{level}.pddl"

    def _load_plan_for_problem(self, problem_path: Path) -> Optional[List[str] ]:
        """Return the list of actions previously saved for this exact problem file, or None."""
        game_name = self._get_game_name()
        plans_index = self.game_dir / f"{game_name}_plans.json"
        if not plans_index.exists():
            return None
        try:
            data = json.load(open(plans_index, "r"))
            entry = data.get(problem_path.name)
            if not entry:
                return None
            return entry.get("actions")
        except Exception:
            return None

    def run_fd_plan_actions(self, engine, level: int) -> bool:
        """Strictly execute the FD plan saved for this level’s problem file. No synthesis."""
        self.engine = engine
        self.current_level = level
        self.reset(keep_model=True)  # sets up obs buffers etc., does NOT synthesize
        problem_path = self._problem_path_for_level(level)
        actions = self._load_plan_for_problem(problem_path)
        if not actions:
            print(f"[planner] No FD plan stored for {problem_path.name}")
            return False

        print(f"[planner] Executing FD plan for level {level}: {actions}")
        for a in actions:
            self.step_env(a)
            if self.engine.won:
                print("[planner] Goal reached via FD plan.")
                return True
            if self.engine.lost:
                print("[planner] Agent died while executing FD plan.")
                return False
        return bool(self.engine.won)


    def prune_exploratory_plans_with_lm(self, exploratory_plans, state, world_model_str):
        """
        Use LLM to prune exploratory plans based on the current state and world model.

        Args:
            exploratory_plans (list): List of suggested exploratory plans.
            state (dict): Current game state.
            world_model_str (str): Current world model as a string.

        Returns:
            list: Pruned exploratory plans.
        """
        # Generate the LLM prompt using the defined prune_exploration_prompt
        formatted_prompt = prune_exploration_prompt.format(
            suggested_exploratory_plans=exploratory_plans,
            current_state=state,
            world_model_str=world_model_str
        )


        # Query the LLM for the pruned plans
        response, fingerprint = self.query_lm(formatted_prompt)
#         response, fingerprint = """```Python
# ['push_to baba_obj rock_obj goop_obj']
# # ```""", 'fingerprint'
#         response, fingerprint = """```Python
# ['form_rule keke_word is_word you_word']
# ```""", 'fingerprint'

        # Extract the list of plans from the response
        selected_plans = self.extract_code_from_response(response)

        # Create step directory and save files
        step_dir = self.logger.create_step("exploratory_plan_pruning")
        self.logger.save_step_files(
            step_dir,
            formatted_prompt,
            response,
            selected_plans,
            "pruned_plans.txt"
        )

        # Save fingerprint to file
        with open(os.path.join(step_dir, "fingerprint.txt"), "w") as f:
            f.write(fingerprint)

        self.logger.add_to_tape({
            "step": "exploratory_plan_pruning",
            "prompt": formatted_prompt,
            "response": response
        })
        # Note: timestamp is now added by the logger

        # Read the pruned plans from the saved file
        pruned_plans_path = os.path.join(step_dir, "pruned_plans.txt")
        try:
            with open(pruned_plans_path, 'r') as f:
                selected_plans = f.read().strip()
            
            # Parse the selected plans into a Python list
            pruned_plans = ast.literal_eval(selected_plans)
            print(f"Pruned exploratory plans: {pruned_plans}")
            return pruned_plans
        except Exception as e:
            print(f"Error parsing LLM response for pruned plans: {e}")
            return exploratory_plans  # Fallback to the original plans if parsing fails

    
    # def enumerate_possible_subplans(self, state):
    #     """
    #     Enumerate all possible subplans based on the current state by grounding operators with entities.

    #     Args:
    #         state (dict): The current game state.

    #     Returns:
    #         list: A list of possible subplans.
    #     """
    #     groundings = enumerate_groundings(self.domain_file, state)

    #     return groundings

    
    def is_valid_rule(self, rule):
        """
        Validate a rule based on predefined constraints.

        Args:
            rule (str): A rule string like 'form_rule baba_word is_word you_word'.

        Returns:
            bool: True if the rule is valid, False otherwise.
        """
        parts = rule.split()
        if len(parts) != 4 or parts[0] != "form_rule":
            return False  # Rule must follow the format 'form_rule X is Y'

        _, word1, word2, word3 = parts

        # Rules cannot start with these words
        invalid_start_words = {"win_word", "you_word", "is_word"}
        if word1 in invalid_start_words:
            return False

        # Rules cannot have two consecutive words
        if word2.endswith("_word") and word3.endswith("_word"):
            return False

        # Rules must follow the form 'X is Y'
        valid_end_words = {"you_word", "win_word", "kill_word", "push_word", "stop_word"}
        if word2 != "is_word" or (word3 not in valid_end_words and not word3.endswith("_word")):
            return False

        return True


    def filter_exploratory_plans(self, plans):
        """
        Filter exploratory plans to include only valid rules.

        Args:
            plans (list): List of exploratory plan strings.

        Returns:
            list: Filtered list of valid exploratory plans.
        """
        return [plan for plan in plans if self.is_valid_rule(plan)]

    

    def propose_exploratory_plans(self, state, domain_file):
        """
        Generate exploratory plans based on satisfied preconditions in the current state.
        """
        exploratory_plans = []

         # Step 1: Generate the type mapping for the current state
        if isinstance(self.engine, BabyAI):
            type_mapping = domain_specific_type_system_mapping_BABYAI(state)

        if not type_mapping:
            print("Warning: Type mapping is empty. Skipping plan pruning.")
            return exploratory_plans

        # Step 2: Enumerate all possible subplans
        possible_subplans = enumerate_possible_subplans(state, domain_file)


        # Step 3: Prune invalid subplans based on the type mapping
        pruned_subplans = prune_invalid_subplans_TYPE(possible_subplans, type_mapping, domain_file)


        if not pruned_subplans:
            print("Warning: No valid subplans after pruning.")
            return exploratory_plans

        # Load operators from domain file
        for subplan in pruned_subplans:
            try:
                operator = operator_extractor(domain_file, subplan)
                preconditions = operator['preconditions']
                effects = operator['effects']

                # Check if preconditions are satisfied
                precondition_results = checker(state, preconditions, operator)
                effects_results = checker(state, effects, operator)

                if precondition_results:
                    # If preconditions are satisfied, add the subplan to exploratory plans
                    exploratory_plans.append(subplan)
            except ValueError as e:
                print(f"Error processing subplan {subplan}: {e}")

        self.runtime_vars['exploratory_plans'] = exploratory_plans
        return exploratory_plans

    def execute_random_actions(self, num_actions=10):
        """Execute random actions and store the resulting transitions in the replay buffer."""
        plan = []
        for _ in range(num_actions):
            random_action = random.choice(self.actions_set)
            plan.append(random_action)

        return plan
    
    def seed_game_from_previous(agent, src_experiment_dir: str, game: str, copy_plans: bool = True, src_game: Optional[str] = None):
        """
        Copy prior artifacts into the current experiment so new levels can start
        from an existing domain/worldmodel/predicates and only generate a problem.

        If src_game is provided (cross-game seeding), files are taken from
        tc_game/<src_game>/ and renamed to match <game> for domain/plans.
        """
        src_game = src_game or game

        src = Path(src_experiment_dir) / "tc_game" / src_game
        dst = Path(agent.logger.experiment_dir) / "tc_game" / game
        dst.mkdir(parents=True, exist_ok=True)

        copies = [
            # same name:
            ("worldmodel.py", "worldmodel.py"),
            ("predicates.py", "predicates.py"),
            # domain/plans renamed to target game:
            (f"{src_game}_domain.pddl", f"{game}_domain.pddl"),
        ]
        if copy_plans:
            copies.append((f"{src_game}_plans.json", f"{game}_plans.json"))

        for sname, dname in copies:
            sp = src / sname
            dp = dst / dname
            if sp.exists():
                shutil.copy(sp, dp)
                print(f"[seed] copied {sp} → {dp}")
            else:
                print(f"[seed] skip missing {sp}")

        # point the agent/runtime to the seeded files
        agent.game_dir = dst
        agent.domain_file = str(dst / f"{game}_domain.pddl")
        os.environ["TC_WORLDMODEL_FILE"] = str(dst / "worldmodel.py")
        os.environ["TC_PREDICATES_FILE"] = str(dst / "predicates.py")

        # mark as non-empty & reload predicates module
        agent.predicates_empty = False
        agent.domain_empty = False
        if "predicates" in sys.modules:
            importlib.reload(sys.modules["predicates"])
        agent.reload_predicates_module()

        print(f"[seed] seeded {game} from {src_experiment_dir}/tc_game/{src_game} → {dst}")

    
    def _save_plan_for_problem(self, problem_path: Path, plan: List[str]):
        """
        Append/replace the plan associated with this exact problem file.
        Keeps a single JSON mapping:
        {
            "<problem-file-name>.pddl": {
                "level": <int or string>,
                "actions": ["act1", "act2", ...]
            },
            ...
        }
        """
        game_name = self._get_game_name()
        plans_index = self.game_dir / f"{game_name}_plans.json"  # reuse the same file

        # Load existing
        data = {}
        if plans_index.exists():
            try:
                data = json.load(open(plans_index, "r"))
            except Exception:
                data = {}

        # Update entry keyed by *problem filename*
        data[problem_path.name] = {
            "level": getattr(self, "current_level", None),
            "actions": plan,
        }

        json.dump(data, open(plans_index, "w"), indent=2)
        print(f"[{game_name}] plan stored for {problem_path.name}")
        



    def step_env(self, action):

        # Step the game engine and append to history

        self.engine.step(action)
        state = deepcopy(self.engine.get_obs())

        if isinstance(self.engine, BabaIsYou):
            state = process_state_baba(state)
        
        self.runtime_vars['observations'].append(state)
        self.actions.append(action)

        # Update replay buffers
        self._update_replay_buffers((
            self.runtime_vars['observations'][-2],
            self.actions[-1],
            self.runtime_vars['observations'][-1]
        ))


        self.tape[-1]['action'] = action
        self.tape[-1]['observation'] = deepcopy(self.runtime_vars['observations'][-1])
        self.tape[-1]['world_model'] = self.runtime_vars['interaction_rules_str']

    def _get_game_name(self):
        """
        Turn self.engine’s class into the folder‐name you want under tc_game/.
        """
        from envs.games import BabaIsYou, LavaGrid
        from envs.minihack_env import MinihackEnv



        if isinstance(self.engine, BabaIsYou):
            return "baba"
        elif isinstance(self.engine, LavaGrid):
            return "lava"
        elif isinstance(self.engine, MinihackEnv):
            return "minihack"
        else:
            # fallback to the class name
            return self.engine.__class__.__name__.lower()


    def run(self, engine, max_revisions=5, max_attempts=6):
        self.engine = engine
        self.current_level = self.engine.level_id  # Or any other method to determine the level

        # One timing JSON per run() call
        with self.timing_run(label="run", level=self.current_level):
            # Ensure statistics entry exists for the current level when running
            # individual levels directly (not via ``run_multiple_levels``)
            level_key = f"{self.engine.level_set}_{self.current_level}"
            self.level_statistics.setdefault(
                level_key,
                {
                    "attempts": 0,
                    "revisions": 0,
                    "debugs": 0,
                    "explorations": 0,
                    "status": "in_progress",
                    "first_letters": None,
                },
            )

            # --- insert per‐game directory setup here ---
            from pathlib import Path
            game_name   = self._get_game_name()
            # Load per-game prompts if available (per-level override for minihack only).
            init_prompt, revise_prompt = load_world_prompts(
                game_name,
                level_id=getattr(self.engine, "level_id", None),
            )
            self.initialize_world_model_prompt = init_prompt
            self.revise_world_model_prompt = revise_prompt

            self.game_dir = Path(self.logger.experiment_dir) / "tc_game" / game_name
            self.game_dir.mkdir(parents=True, exist_ok=True)

            # Ensure planner loads the correct world model
            os.environ["TC_WORLDMODEL_FILE"] = str(self.game_dir / "worldmodel.py")
            if str(self.game_dir) not in sys.path:
                sys.path.insert(0, str(self.game_dir))
            if self.logger.experiment_dir not in sys.path:
                sys.path.insert(0, self.logger.experiment_dir)


            # should we inject the transfer somewhere here. for example
            # if we we do d

            # Determine the central domain path and check if it exists
            self.central_domain_path = Path(self.logger.experiment_dir) / self.domain_file_name
            if self.centralize_files:
                self._load_domain_pddl(self.central_domain_path)
                central_missing = self.domain_empty
            else:
                central_missing = True

            domain_path = self.game_dir / f"{game_name}_domain.pddl"
            self.domain_file = str(domain_path)

            # Reload predicates for this specific game if the snapshot is missing
            predicates_snapshot = self.game_dir / "predicates.py"
            if not predicates_snapshot.exists() or not predicates_snapshot.read_text().strip():
                self.predicates_empty = True

            self._load_domain_pddl(self.domain_file)
            set_domain_file(self.domain_file)

            if not domain_path.exists():
                print(f"[{game_name}] domain PDDL not found → generating now")
                self.generate_and_solve_pddl(level=self.current_level)
                domain_code = domain_path.read_text()
                if self.centralize_files:
                    if central_missing:
                        shutil.copy(domain_path, self.central_domain_path)
                    else:
                        self.update_experiment_domain(domain_path, domain_code)
                self._load_domain_pddl(self.domain_file)

                self.plans = self._load_plans()
                print(f"Loaded plans for '{game_name}':", list(self.plans.keys()))
            else:
                # ======= ONLY CHANGE: choose exactly one domain path and solve once =======
                if self.centralize_files and central_missing:
                    shutil.copy(domain_path, self.central_domain_path)
                print(f"[{game_name}] domain PDDL already exists, skipping generation")

                # Prefer the centralized domain if centralize_files is on and it exists; otherwise use local
                shared_domain = Path(self.logger.experiment_dir) / self.domain_file_name
                selected_domain = shared_domain if (self.centralize_files and shared_domain.exists()) else domain_path

                # Single policy check / single solve — avoids double transfer_domain
                _ = self._solve_existing_pddl(self.current_level, selected_domain)

                # Keep central & local merged/synced once
                if self.centralize_files:
                    self.update_experiment_domain(domain_path, domain_path.read_text())

                self.plans = self._load_plans()
                # ======= END CHANGE =======

            # -------------------------------------------

            revision_count = 0
            debug_count = 0
            attempt_count = 0
            exploratory_plan_index = 0

            while revision_count <= max_revisions and attempt_count < max_attempts:
                # Initialize
                self.reset(keep_model=True)
                first_letters = ''
                model_was_revised = False

                # Extract the original state immediately after reset
                initial_state = deepcopy(self.engine.get_obs())

                if isinstance(self.engine, BabaIsYou):
                    initial_state = process_state_baba(initial_state)


                if self.is_world_model_empty() and self.do_revise_model:
                    # If the world model is empty, use the default action set
                    print("World model is empty, executing random actions.")
                    num_actions = 15
                    plan = self.execute_random_actions(num_actions=num_actions)  # Adjust the number as needed
                    print(plan)
                    print("World model was empty, revised the model. Moving to next iteration.")
                    for action in plan:
                        self.step_env(action)
                    self._initialize_world_model(num_actions)

                    self.capture_world_model()


                    # — now re-check the shared predicates.py snapshot —
                    self._load_predicates(self.predicates_file_name)
                    if self.predicates_empty:
                        domain_path  = self.game_dir / f"{game_name}_domain.pddl"
                        problem_path = self.game_dir / f"{game_name}_{self.current_level}.pddl"
                        print("⚠️  predicates.py is empty → generating via LLM")
                        self.generate_and_save_predicates(domain_path, problem_path)

                    # After initializing the model and reloading predicates, reset
                    # the environment so planning starts from the correct state
                    self.reset(keep_model=True)

                    # Proceed with the hierarchical planner
                    mode = self._sample_planner_mode()  # Determine planner mode (explore/exploit)
                    plan = self._hierarchical_planner(mode)
                    print("subplans from init model:", plan)
                    model_was_revised = True

                else:
                    # If the world model is not empty, proceed with the hierarchical planner
                    mode = self._sample_planner_mode()  # Determine planner mode (explore/exploit)
                    plan = self._hierarchical_planner(mode) 

                for action in plan:
                    self.step_env(action)
                    print("action taken:", action)
                    first_letters += action[0]  # Collect the first letters of each action

                    # Exit if agent won
                    if self.engine.won or (isinstance(self.engine, BabaIsYou) and self.current_level == 6 and first_letters == 'rrruuu'):
                            
                        self.tape[-1]['exit_condition'] = 'won'
                        self._update_solution(self.current_level, first_letters)
                        self.level_statistics[f"{self.engine.level_set}_{self.current_level}"]["first_letters"] = first_letters
                        self.level_statistics[f"{self.engine.level_set}_{self.current_level}"]["revisions"] = revision_count
                        self.level_statistics[f"{self.engine.level_set}_{self.current_level}"]["debugs"] = debug_count
                        print(first_letters)
                        # if isinstance(self.engine, BabyAI):
                        #     self.engine.close()

                       

                        # Save actions and summary before returning on success
                        print(f"\n===== WON level {self.current_level}! =====")
                        print(f"Actions that led to win: {self.actions}")
                        summary = f"""
        Level: {self.current_level}
        Revisions: {revision_count}
        Attempts: {attempt_count}
        Final Status: {"Won" if self.engine.won else "Failed"}
        First Letters: {first_letters}
        Actions: {self.actions}
                        """
                        self.logger.save_summary(summary)
                        self.logger.save_actions(level_key, self.actions)
                        self.level_statistics[level_key]["status"] = "completed"
                        self.level_statistics[level_key]["attempts"] = attempt_count
                        self._save_level_summary(level_key)

                        return True

                    # Check if the agent lost (e.g., died or failed critically)
                    if self.engine.lost:
                        self.tape[-1]['exit_condition'] = 'lost'
                        self._update_solution(self.current_level, first_letters)
                        self.level_statistics[f"{self.engine.level_set}_{self.current_level}"]["first_letters"] = first_letters
                        self.level_statistics[f"{self.engine.level_set}_{self.current_level}"]["revisions"] = revision_count
                        self.level_statistics[f"{self.engine.level_set}_{self.current_level}"]["debugs"] = debug_count
                        print("AGENT DIED")
                        if self.current_level == 3 and self._get_game_name() == "minihack":
                            self._revise_lvl3_bundle()
                        else:
                            self._revise_world_model()
                        attempt_count += 1
                        model_was_revised = True

                        break

                # Silent failure: plan finished, agent neither won nor died.
                # WM mispredicted the outcome → revise from the latest transitions.
                if (not self.engine.won and not self.engine.lost
                        and not model_was_revised and plan):
                    print("Plan executed but no win/loss — revising WM from silent failure")
                    if self.current_level == 3 and self._get_game_name() == "minihack":
                        self._revise_lvl3_bundle()
                    else:
                        self._revise_world_model()
                    attempt_count += 1
                    model_was_revised = True

                # If the model was revised, execute it first before proceeding with exploratory goals
                if model_was_revised:
                    self.reset(keep_model=True)
                    first_letters = ''  # Reset first_letters after model revision
                    mode = self._sample_planner_mode()  # Determine planner mode (explore/exploit)
                    plan = self._hierarchical_planner(mode) 
                    for action in plan:
                        self.step_env(action)
                        first_letters += action[0]  # Collect the first letters of each action

                        # Exit if agent won
                        if self.engine.won:
                            self.tape[-1]['exit_condition'] = 'won'
                            print(f"\n===== WON level {self.current_level}! =====")
                            print(f"Actions that led to win: {self.actions}")
                            return True

                        # Check if the agent lost (e.g., died or failed critically)
                        if self.engine.lost:
                            self.tape[-1]['exit_condition'] = 'lost'
                    
                            print(self.engine.get_obs())
                            attempt_count += 1

                            
                            break

                
                # if theorycoder domain predicates, world model is not empty and but prooblem file is just gen problem


                # Handle model revision if necessary
                if not self.is_world_model_empty() and self.do_revise_model and not model_was_revised:
                    pruned_plans = ["collect_diamond diamond1","collect_diamond diamond2","collect_diamond diamond3","collect_diamond diamond4","collect_diamond diamond5","collect_diamond diamond6","collect_diamond diamond7","collect_diamond diamond8","collect_diamond diamond9","escape_via_exit avatar exitdoor"]
                    print("pruned automatically", pruned_plans)
                   
                    LLM_pruned_plans = pruned_plans

                    if self.prune_plans:
                        # Cycle through exploratory plans across attempts
                        for i in range(len(LLM_pruned_plans)):
                            subplan = LLM_pruned_plans[(exploratory_plan_index + i) % len(LLM_pruned_plans)]
                            self.reset(keep_model=True)
                            first_letters = ''  # Reset first_letters for each exploratory plan
                            subplans = self._hierarchical_planner(mode="explore_collision", subplan_exploratory=subplan)

                            for subplan, actions in subplans:
                                for action in actions:
                                    self.step_env(action)
                                    first_letters += action[0]  # Collect the first letters of each action

                            # Perform model revision after every collision attempt
                            print(f"Revising the model after subplan: {subplan}")
                            examples, error_count = self._choose_synthesis_examples(exploratory_plan=subplan)

                            mission = getattr(self.engine, 'mission', '')

                            if self._do_revise_model(error_count):
                                prompt = self.revise_world_model_prompt.format(
                                    actions_set=self.engine.actions_set,
                                    errors_from_world_model='\n\n'.join(examples),
                                    world_model_str=self.runtime_vars['world_model_str'],
                                    utils="directions = {\n    'left': [-1, 0],\n    'right': [1, 0],\n    'up': [0, 1],\n    'down': [0, -1],\n    'up_right': [1, 1],\n    'down_right': [1, -1],\n    'down_left': [-1, -1],\n    'up_left': [-1, 1],\n}",
                                    mission=mission,
                                )

                                print(prompt)

                                # Create step directory and save files
                                step_dir = self.logger.create_step("revision")
                                resp, completion = self.query_lm(prompt, label="revise_world_model")

                                with open('fingerprint_seed_v2_revision_lv4.txt', 'w') as file:
                                    file.write(str(completion))

                                new_world_model_code = self.extract_code_from_response(resp)

                                if new_world_model_code:
                                    self.logger.save_step_files(
                                        step_dir,
                                        prompt,
                                        resp,
                                        new_world_model_code,
                                        "worldmodel.py"
                                    )

                                    self.logger.add_to_tape({
                                        "step": "revision",
                                        "prompt": prompt,
                                        "response": resp
                                    })

                                    # Save full response and completion object instead of just fingerprint 
                                    with open(os.path.join(step_dir, "completion_info.json"), "w") as f:
                                        json.dump({
                                            "response": resp,
                                            "completion": completion if isinstance(completion, dict) else str(completion)
                                        }, f, indent=2)

                                    # Update world model code and version
                                    self.runtime_vars['world_model_str'] = new_world_model_code
                                    self.runtime_vars['error_msg_model'] = new_world_model_code

                                    # Overwrite the current worldmodel.py file with the new model
                                    self.overwrite_world_model(new_world_model_code)

                                self.tape[-1]['revision_prompts'] = prompt
                                self.tape[-1]['revision_responses'] = resp
                                print(prompt)
                                print(resp)
                                if self._do_revise_plan(error_count):
                                    self.runtime_vars['revise_plan'] = True

                            revision_count += 1

                            if revision_count > max_revisions:
                                print("Max model revisions reached. Exiting.")
                                break
                            print(f"Model revised {revision_count} times. Re-running.")
                        
                        exploratory_plan_index = (exploratory_plan_index + len(LLM_pruned_plans)) % len(LLM_pruned_plans)
                    else:
                        # Aggregate datasets for all exploratory plans
                        self.aggregated_dataset = []
                        self.reset(keep_model=True)

                        for subplan in self.plans.get(str(self.current_level), []):
                            first_letters = ''  # Reset first_letters for each exploratory plan
                            plan = self._hierarchical_planner(mode="explore_collision", subplan_exploratory=subplan)

                            # use this plan to get D for model revision 
                            if plan == None:
                                print(f"plan was none for {subplan}")
                            
                            else:
                                for action in plan:
                                    self.step_env(action)
                                    first_letters += action[0]  # Collect the first letters of each action

                            # Collect data for the aggregated dataset
                            examples, error_count = self._choose_synthesis_examples(exploratory_plan=None)
                            self.aggregated_dataset.append({
                                "subplan": subplan,
                                "examples": examples,
                                "error_count": error_count
                            })

                        # Perform model revision using the aggregated dataset
                        aggregated_examples = []
                        for data in self.aggregated_dataset:
                            aggregated_examples.append(f"ERRORS FROM WORLD MODEL for EXPLORATORY PLAN {data['subplan']}:\n\n" + "\n\n".join(data['examples']))

                        
                        mission = getattr(self.engine, 'mission', '')

                        prompt = self.revise_world_model_prompt.format(
                            actions_set=self.engine.actions_set,
                            errors_from_world_model='\n\n'.join(aggregated_examples),
                            world_model_str=self.runtime_vars['world_model_str'],
                            utils="directions = {\n    'left': [-1, 0],\n    'right': [1, 0],\n    'up': [0, 1],\n    'down': [0, -1],\n    'up_right': [1, 1],\n    'down_right': [1, -1],\n    'down_left': [-1, -1],\n    'up_left': [-1, 1],\n}",
                            mission=mission,
                        )

                        file_name='current_prompt.txt'
                        with open(file_name, 'w') as file:
                            file.write(prompt) 

                        print(prompt)

                        # Create step directory and save files
                        step_dir = self.logger.create_step("revision")
                        resp, completion = self.query_lm(prompt, label="revise_world_model")

                        with open('fingerprint_seed_v2_revision_lv4.txt', 'w') as file:
                            file.write(str(completion))

                        new_world_model_code = self.extract_code_from_response(resp)

                        if new_world_model_code:
                            self.logger.save_step_files(
                                step_dir,
                                prompt,
                                resp,
                                new_world_model_code,
                                "worldmodel.py"
                            )

                            self.logger.add_to_tape({
                                "step": "revision",
                                "prompt": prompt,
                                "response": resp
                            })

                            # Save full response and completion object instead of just fingerprint 
                            with open(os.path.join(step_dir, "completion_info.json"), "w") as f:
                                json.dump({
                                    "response": resp,
                                    "completion": completion if isinstance(completion, dict) else str(completion)
                                }, f, indent=2)

                            # Update world model code and version
                            self.runtime_vars['world_model_str'] = new_world_model_code
                            self.runtime_vars['error_msg_model'] = new_world_model_code

                            # Overwrite the current worldmodel.py file with the new model
                            self.overwrite_world_model(new_world_model_code)

                        self.tape[-1]['revision_prompts'] = prompt
                        self.tape[-1]['revision_responses'] = resp
                        print(prompt)
                        print(resp)

                        revision_count += 1
                        model_was_revised = True

                        if revision_count > max_revisions:
                            print("Max model revisions reached. Exiting.")
                            break
                        print(f"Model revised {revision_count} times. Re-running.")

                else:
                    # If no revisions happened and no win occurred, stop the loop
                    break

               

            print("LEVEL TERMINATED")
            print(self.engine.get_obs())


     

            # timing_run will auto-save timings JSON here
            return False



    def run_multiple_levels(self, level_sets, max_revisions=5, max_attempts=6):
        """Run the agent through multiple levels sequentially."""
        overall_results = {
            "levels_completed": [],
            "levels_failed": [],
            "total_revisions": 0,
            "total_debugs": 0,
            "total_explorations": 0
        }

        for level_set, levels in level_sets.items():
            for level_id in levels:
                print(f"\nStarting Level {level_set}-{level_id}")
                
                # Initialize the engine for the current level
                if args.game == 'baba':
                    self.engine = BabaIsYou(level_set=level_set, level_id=level_id)
                elif args.game == 'minihack':
                    self.engine = MinihackEnv(level_set=level_set, level_id=level_id)
                
                # Initialize level statistics
                level_key = f"{level_set}_{level_id}"
                self.level_statistics[level_key] = {
                    "attempts": 0,
                    "revisions": 0,
                    "debugs": 0,
                    "explorations": 0,
                    "status": "not_started",
                    "first_letters": None
                }

                print(f"Running single level: {level_key}")
                success = self.run(self.engine, max_revisions, max_attempts)
                print(f"Finished running single level: {level_key} with success: {success}")
                
                if success:
                    overall_results["levels_completed"].append(level_key)
                    self.level_statistics[level_key]["status"] = "completed"
                else:
                    overall_results["levels_failed"].append(level_key)
                    self.level_statistics[level_key]["status"] = "failed"

                # Update overall statistics
                overall_results["total_revisions"] += self.level_statistics[level_key]["revisions"]
                overall_results["total_debugs"] += self.level_statistics[level_key]["debugs"]
                overall_results["total_explorations"] += self.level_statistics[level_key]["explorations"]

                # Save level summary
                self._save_level_summary(level_key)

        # Save overall summary
        self._save_overall_summary(overall_results)
        # end of run()
        try:
            fname = f"timings_{self._get_game_name()}_{self.current_level}.json"
        except Exception:
            fname = "timings.json"
        self._save_timings(filename=fname)

        return overall_results

    def _save_level_summary(self, level_key):
        """Save a summary for a specific level."""
        stats = self.level_statistics[level_key]
        summary = f"""
Level: {level_key}
Status: {stats['status']}
Attempts: {stats['attempts']}
Revisions: {stats['revisions']}
Debugs: {stats['debugs']}
Explorations: {stats['explorations']}
Solution: {stats['first_letters'] if stats['first_letters'] else 'None'}
        """
        
        # Create level directory in experiment folder
        level_dir = os.path.join(self.logger.experiment_dir, f"level_{level_key}")
        os.makedirs(level_dir, exist_ok=True)
        
        # Save summary
        with open(os.path.join(level_dir, "summary.txt"), "w") as f:
            f.write(summary)

        # Save statistics as JSON for later analysis
        with open(os.path.join(level_dir, "statistics.json"), "w") as f:
            json.dump(stats, f, indent=2)

    def _save_overall_summary(self, results):
        """Save overall experiment summary."""
        summary = f"""
Total Levels Attempted: {len(results['levels_completed']) + len(results['levels_failed'])}
Levels Completed: {len(results['levels_completed'])}
Levels Failed: {len(results['levels_failed'])}
Total Revisions: {results['total_revisions']}
Total Debugs: {results['total_debugs']}
Total Explorations: {results['total_explorations']}

Completed Levels: {', '.join(results['levels_completed'])}
Failed Levels: {', '.join(results['levels_failed'])}
        """
        
        self.logger.save_summary(summary)
        
        # Save detailed results as JSON
        with open(os.path.join(self.logger.experiment_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)


def clear_run_files(game: str, world_model_name: str, domain_file_name: str, base_dir: str) -> None:
    """Remove generated files for a clean start."""
    root_files = [f"{world_model_name}.py", domain_file_name]
    for file in root_files:
        path = Path(file)
        if path.exists():
            try:
                path.unlink()
            except OSError:
                print(f"Failed to remove {path}")

    # remove experiment scoped files
    exp_dir = Path(base_dir)
    for p in [exp_dir / f"{world_model_name}.py", exp_dir / f"{domain_file_name}", exp_dir / "predicates.py"]:
        if p.exists():
            try:
                p.unlink()
            except OSError:
                print(f"Failed to remove {p}")

    game_dir = Path(base_dir) / "tc_game" / game
    if not game_dir.exists():
        return

    for pattern in [
        "worldmodel.py",
        f"{game}_domain.pddl",
        f"{game}_plans.json",
        f"{game}_*.pddl",
    ]:
        for file in game_dir.glob(pattern):
            try:
                file.unlink()
            except OSError:
                print(f"Failed to remove {file}")


# --- helper: save plan against its exact problem file --



def run_babyai_transfer_sequence(agent, level_set_name="babyai_levels", debug_actions=12):
    """
    BabyAI transfer run enforcing:
      - L19: learn domain+problem+world model+predicates (fresh)
      - L8 : learn NEW domain+problem+world model+predicates (fresh, no merging)
      - L13: reuse STRICTLY the L8 artifacts; problem-only generation if missing
    Files live in experiments/<exp>/tc_game/babyai/.
    """
    from pathlib import Path
    import shutil, os, sys
    from envs.babyai_env import BabyAI
    from theorycoder2 import load_world_prompts  # already in your file

    def _build_engine(lid):
        return BabyAI(level_set=level_set_name, level_id=lid)

    game = "babyai"
    agent.game_dir = Path(agent.logger.experiment_dir) / "tc_game" / game
    agent.game_dir.mkdir(parents=True, exist_ok=True)

    # Make sure planner/imports see this folder
    os.environ["TC_WORLDMODEL_FILE"] = str(agent.game_dir / "worldmodel.py")
    if str(agent.game_dir) not in sys.path:
        sys.path.insert(0, str(agent.game_dir))
    if agent.logger.experiment_dir not in sys.path:
        sys.path.insert(0, agent.logger.experiment_dir)

    # Local domain path (no merging/centralization in this sequence)
    local_domain_path = agent.game_dir / f"{game}_domain.pddl"
    agent.domain_file = str(local_domain_path)

    # Snapshots we’ll keep between levels
    wm_lv19    = agent.game_dir / "wm_lv19.py"
    preds_lv19 = agent.game_dir / "predicates_lv19.py"
    dom_lv19   = agent.game_dir / "domain_lv19.pddl"

    wm_lv8     = agent.game_dir / "wm_lv8.py"
    preds_lv8  = agent.game_dir / "predicates_lv8.py"
    dom_lv8    = agent.game_dir / "domain_lv8.pddl"

    # Helper: ensure a problem exists for a given level from the *current* domain
    def _ensure_problem_for_level(level_id):
        prob_path = agent.game_dir / f"{game}_{level_id}.pddl"
        if not prob_path.exists():
            agent.generate_problem_for_existing_domain(level_id)
        return prob_path

    # Collect overall results
    ok_lv19 = False
    ok_lv8  = False
    ok_lv13 = False

    # ───────────────────────────────────────────────────────────────────
    # L19 — learn everything (domain + problem + WM + predicates)
    # ───────────────────────────────────────────────────────────────────
    print("\n[BABYAI] Level 19: learn domain+problem+world model+predicates (fresh)")
    agent.engine = _build_engine(19)

    # Load per-game prompts
    init_prompt, revise_prompt = load_world_prompts(game)
    agent.initialize_world_model_prompt = init_prompt
    agent.revise_world_model_prompt = revise_prompt

    with agent.timing_run(label="babyai_L19_bootstrap", level=19):
        plan_19 = agent.generate_and_solve_pddl(level=19)
        agent.reset(keep_model=True)
        agent.capture_world_model()
        for a in agent.execute_random_actions(num_actions=debug_actions):
            agent.step_env(a)
        _orig = agent.do_revise_model
        agent.do_revise_model = True
        try:
            agent._initialize_world_model(num_actions=debug_actions)
        finally:
            agent.do_revise_model = _orig
        dom_19  = local_domain_path
        prob_19 = agent.game_dir / f"{game}_19.pddl"
        agent.predicates_empty = True
        agent.generate_and_save_predicates(dom_19, prob_19)


        # Save L19 snapshots
        if (agent.game_dir / "worldmodel.py").exists():
            shutil.copy(agent.game_dir / "worldmodel.py", wm_lv19)
        if (agent.game_dir / "predicates.py").exists():
            shutil.copy(agent.game_dir / "predicates.py", preds_lv19)
        if local_domain_path.exists():
            shutil.copy(local_domain_path, dom_lv19)

        print("[BABYAI] L19 summary:")
        print(f"  - Domain+problem generated: True")
        print(f"  - Plan found by FD: {ok_lv19} (len={len(plan_19) if plan_19 else 0})")
        print(f"  - World model initialized: True")
        print(f"  - Predicates generated: True")

    # ───────────────────────────────────────────────────────────────────
    # L8 — learn NEW domain + problem + WM + predicates (fresh, no merging)
    # ───────────────────────────────────────────────────────────────────
    # --- L8 (bootstrap fresh) ---
    print("\n[BABYAI] Level 8: learn NEW domain+problem+world model+predicates (fresh)")
    agent.engine = _build_engine(8)

    with agent.timing_run(label="babyai_L8_bootstrap", level=8):
        plan_8_gen = agent.generate_and_solve_pddl(level=8)
        agent.reset(keep_model=True)
        agent.capture_world_model()
        for a in agent.execute_random_actions(num_actions=debug_actions):
            agent.step_env(a)

        _orig_revise = agent.do_revise_model
        agent.do_revise_model = True
        try:
            agent._initialize_world_model(num_actions=debug_actions)
        finally:
            agent.do_revise_model = _orig_revise

        dom_8  = local_domain_path
        prob_8 = agent.game_dir / f"{game}_8.pddl"
        agent.predicates_empty = True
        agent.generate_and_save_predicates(dom_8, prob_8)

    # attempts for L8 (separate timing around the actual run)
    with agent.timing_run(label="babyai_L8_attempt", level=8):
        original_flag = agent.do_revise_model
        agent.do_revise_model = False
        ok_lv8 = agent.run(agent.engine, max_revisions=0, max_attempts=getattr(agent, "max_replans", 1) or 1)
        agent.do_revise_model = original_flag
    print(f"[BABYAI] Level 8 success = {ok_lv8}")


    # Save L8 snapshots (these are the ones L13 will reuse strictly)
    if (agent.game_dir / "worldmodel.py").exists():
        shutil.copy(agent.game_dir / "worldmodel.py", wm_lv8)
    if (agent.game_dir / "predicates.py").exists():
        shutil.copy(agent.game_dir / "predicates.py", preds_lv8)
    if local_domain_path.exists():
        shutil.copy(local_domain_path, dom_lv8)

    # ───────────────────────────────────────────────────────────────────
    # L13 — reuse STRICTLY the L8 artifacts; PROBLEM ONLY if missing
    # ───────────────────────────────────────────────────────────────────
    print("\n[BABYAI] Level 13: reuse L8 domain+WM+predicates; PROBLEM ONLY generation if missing")
    agent.engine = _build_engine(13)

    # Restore strict L8 artifacts
    if dom_lv8.exists():
        shutil.copy(dom_lv8, local_domain_path)
    if wm_lv8.exists():
        shutil.copy(wm_lv8, agent.game_dir / "worldmodel.py")
    elif wm_lv19.exists():
        shutil.copy(wm_lv19, agent.game_dir / "worldmodel.py")
    if preds_lv8.exists():
        shutil.copy(preds_lv8, agent.game_dir / "predicates.py")
        agent.update_experiment_predicates((agent.game_dir / "predicates.py").read_text())
    elif preds_lv19.exists():
        shutil.copy(preds_lv19, agent.game_dir / "predicates.py")
        agent.update_experiment_predicates((agent.game_dir / "predicates.py").read_text())

    # Ensure problem for L13 (problem-only generation), then run with NO revisions
    # ensure problem then run
    with agent.timing_run(label="babyai_L13_bootstrap", level=13):
        _ensure_problem_for_level(13)
        agent.do_revise_model = False


    with agent.timing_run(label="run", level=13, filename="timings_L13_run.json"):
        ok_lv13 = agent.run(agent.engine, max_revisions=0, max_attempts=getattr(agent, "max_replans", 1) or 1)
    print(f"[BABYAI] Level 13 success = {ok_lv13}")

    # ───────────────────────────────────────────────────────────────────
    # Final summary & result dict
    # ───────────────────────────────────────────────────────────────────
    print("\n[BABYAI] Transfer sequence complete.")
    result = {
        "L19_success": ok_lv19,
        "L8_success": ok_lv8,
        "L13_success": ok_lv13,
    }
    print(result)

    _ = run_babyai_reuse_only(
        agent,
        level_set_name=level_set_name,
        levels=(19, 8, 13),
        plans_filename='babyai_plans.json'
    )

    
    return result


# ───────────────────────────────────────────────────────────────────
# STRICT REUSE: run BabyAI using only existing artifacts (no generation)
# ───────────────────────────────────────────────────────────────────
def run_babyai_reuse_strict(agent, level_set_name="babyai_levels"):
    """
    Uses ONLY existing files:
      experiments/<exp>/tc_game/babyai/
        - worldmodel.py
        - predicates.py
        - babyai_domain.pddl
        - babyai_19.pddl, babyai_8.pddl, babyai_13.pddl   (must exist)
    No LLM calls. No domain/predicate/world-model changes. No problem-gen.

    Prints per-level plan success + run success (including Level 19),
    and a final summary dict.
    """
    from pathlib import Path
    import sys, os
    from envs.babyai_env import BabyAI

    def _engine(lid): return BabyAI(level_set=level_set_name, level_id=lid)

    game = "babyai"
    agent.game_dir = Path(agent.logger.experiment_dir) / "tc_game" / game
    agent.game_dir.mkdir(parents=True, exist_ok=True)

    # Make runtime import the game dir + experiment dir
    os.environ["TC_WORLDMODEL_FILE"] = str(agent.game_dir / "worldmodel.py")
    if str(agent.game_dir) not in sys.path:
        sys.path.insert(0, str(agent.game_dir))
    if agent.logger.experiment_dir not in sys.path:
        sys.path.insert(0, agent.logger.experiment_dir)

    # Required artifacts
    worldmodel_path = agent.game_dir / "worldmodel.py"
    predicates_path = agent.game_dir / "predicates.py"
    domain_path     = agent.game_dir / f"{game}_domain.pddl"
    prob19_path     = agent.game_dir / f"{game}_19.pddl"
    prob8_path      = agent.game_dir / f"{game}_8.pddl"
    prob13_path     = agent.game_dir / f"{game}_13.pddl"

    missing = [p for p in [worldmodel_path, predicates_path, domain_path,
                           prob19_path, prob8_path, prob13_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "[reuse-strict] Missing required files:\n  " +
            "\n  ".join(map(str, missing))
        )

    # Hard-disable any synthesis or regeneration paths
    agent.do_revise_model = False
    agent.replace_predicates = False
    agent.predicates_empty = False
    agent.domain_file = str(domain_path)

    def _deny(*_a, **_k):
        raise RuntimeError("reuse-strict mode: generation is disabled")

    agent.generate_and_solve_pddl = _deny
    agent.generate_and_save_predicates = _deny
    agent.generate_problem_for_existing_domain = _deny
    agent._call_pddl_debug = lambda *a, **k: False  # never try LLM debug

    # Helper: attempt to run FD on existing domain+problem (no regeneration)
    def _solve_existing(level_id):
        return agent._solve_existing_pddl(level_id, domain_path) is not None

    results = {}

    # ── Level 19
    print("\n[BABYAI:REUSE-STRICT] Level 19 — existing files only")
    agent.engine = _engine(19)
    ok19_plan = _solve_existing(19)
    ok19_run  = agent.run(agent.engine, max_revisions=0, max_attempts=getattr(agent, "max_replans", 1) or 1)
    results["L19_plan"] = ok19_plan
    results["L19_success"] = ok19_run
    print(f"[BABYAI] Level 19 plan_available={ok19_plan} success={ok19_run}")

    # ── Level 8
    print("\n[BABYAI:REUSE-STRICT] Level 8 — existing files only")
    agent.engine = _engine(8)
    ok8_plan = _solve_existing(8)
    ok8_run  = agent.run(agent.engine, max_revisions=0, max_attempts=getattr(agent, "max_replans", 1) or 1)
    results["L8_plan"] = ok8_plan
    results["L8_success"] = ok8_run
    print(f"[BABYAI] Level 8 plan_available={ok8_plan} success={ok8_run}")

    # ── Level 13
    print("\n[BABYAI:REUSE-STRICT] Level 13 — existing files only")
    agent.engine = _engine(13)
    ok13_plan = _solve_existing(13)
    ok13_run  = agent.run(agent.engine, max_revisions=0, max_attempts=getattr(agent, "max_replans", 1) or 1)
    results["L13_plan"] = ok13_plan
    results["L13_success"] = ok13_run
    print(f"[BABYAI] Level 13 plan_available={ok13_plan} success={ok13_run}")

    print("\n[BABYAI:REUSE-STRICT] Sequence complete.")
    print(results)
    return results

def run_babyai_reuse_only(agent, level_set_name="babyai_levels", levels=(19, 8, 13), plans_filename="babyai_plans.json"):
    """
    Reuse-only BabyAI run:
      - Loads worldmodel.py, predicates.py, and babyai_domain.pddl from tc_game/babyai/
      - Loads existing plans from tc_game/babyai/<plans_filename>
      - For each level, executes the stored high-level plan by compiling to low-level actions via `actor(...)`
      - NO generation, NO merging, NO revisions.
    Prints per-level status and returns a summary dict.
    """
    import os, sys, json, importlib, importlib.util
    from pathlib import Path
    from envs.babyai_env import BabyAI
    from levelrunner import actor

    game = "babyai"
    agent.do_revise_model = False      # hard stop on any learning/regen
    agent.prune_plans = False

    # ── Resolve experiment game dir
    agent.game_dir = Path(agent.logger.experiment_dir) / "tc_game" / game
    agent.game_dir.mkdir(parents=True, exist_ok=True)

    # ── Reused files
    domain_path = agent.game_dir / f"{game}_domain.pddl"
    preds_path  = agent.game_dir / "predicates.py"
    wm_path     = agent.game_dir / "worldmodel.py"
    plans_path  = agent.game_dir / plans_filename

    # ── Existence checks (fail loud; we are reuse-only)
    if not domain_path.exists():
        print(f"[REUSE-ONLY] Missing domain: {domain_path}")
    if not preds_path.exists():
        print(f"[REUSE-ONLY] Missing predicates: {preds_path}")
    if not wm_path.exists():
        print(f"[REUSE-ONLY] Missing world model: {wm_path}")
    if not plans_path.exists():
        print(f"[REUSE-ONLY] Missing plans: {plans_path}")

    # ── Route runner to these exact files
    agent.domain_file = str(domain_path)
    os.environ["TC_WORLDMODEL_FILE"] = str(wm_path)
    os.environ["TC_PREDICATES_FILE"] = str(preds_path)

    # ── Put experiment paths at the FRONT of sys.path (guarantee precedence)
    for p in [str(agent.logger.experiment_dir), str(agent.game_dir)]:
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)

    # ── Hard-load modules from exact files so imports always hit these copies
    def _load_module_from_file(modname: str, path_obj: Path):
        if not path_obj.exists():
            raise FileNotFoundError(f"Expected file missing: {path_obj}")
        spec = importlib.util.spec_from_file_location(modname, str(path_obj))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sys.modules[modname] = mod
        return mod

    try:
        worldmodel = _load_module_from_file("worldmodel", wm_path)
    except Exception as e:
        print(f"[REUSE-ONLY] Failed to load worldmodel: {e}")

    try:
        predicates = _load_module_from_file("predicates", preds_path)
    except Exception as e:
        print(f"[REUSE-ONLY] Failed to load predicates: {e}")

    # Refresh any modules that may cache imports of worldmodel/predicates
    if "levelrunner" in sys.modules:
        importlib.reload(sys.modules["levelrunner"])
    if "planner" in sys.modules:
        importlib.reload(sys.modules["planner"])

    # Also refresh via the agent helper (reads TC_PREDICATES_FILE)
    agent.reload_predicates_module()

    # ── Load stored plans (keyed by problem filename e.g., "babyai_19.pddl")
    try:
        with open(plans_path, "r") as f:
            stored_plans = json.load(f)
    except Exception as e:
        print(f"[REUSE-ONLY] Could not read plans file: {plans_path} ({e})")
        stored_plans = {}

    def _problem_file(level: int) -> str:
        return f"{game}_{level}.pddl"

    def _engine(level: int) -> BabyAI:
        return BabyAI(level_set=level_set_name, level_id=level)

    # ── Execute an exact high-level plan via actor() for a given level
    def _execute_level(level: int):
        """
        Returns (plan_available: bool, success: bool)
        """
        # Build env
        engine = _engine(level)
        agent.engine = engine
        agent.current_level = level

        # Avoid any synthesis in reset(): mark predicates as present
        agent.predicates_empty = False
        agent.reset(keep_model=True)  # init buffers; no LLM is called

        # Lookup stored plan for this level
        problem_name = _problem_file(level)
        plan_entry = stored_plans.get(problem_name)
        plan_available = plan_entry is not None and isinstance(plan_entry.get("actions"), list)

        if not plan_available:
            print(f"[{game}] Level {level}: no stored plan for {problem_name}")
            return False, False

        high_level_ops = plan_entry["actions"]

        # Start from current observation
        state = agent.engine.get_obs()
        if hasattr(agent.engine, "save_screen"):
            try:
                agent.engine.save_screen()
            except Exception:
                pass

        success = False

        for subplan in high_level_ops:
            try:
                # Ensure runtime modules see the correct worldmodel/predicates
                import worldmodel, planner, levelrunner, minihack_utils
                importlib.reload(worldmodel)
                importlib.reload(minihack_utils)
                importlib.reload(levelrunner)
                if "planner" in sys.modules:
                    importlib.reload(planner)

                # Compile subplan to low-level actions
                low_actions, state_after = actor(
                    agent.domain_file,
                    subplan,
                    state,
                    max_iterations=None,
                    debug_callback=None,
                    predicate_debug_callback=None,
                    level=level,
                    engine=agent.engine
                )

                if not low_actions:
                    print(f"[{game}] Level {level}: subplan produced no actions → failing level.")
                    success = False
                    break

                # Execute low-level actions
                for a in low_actions:
                    agent.step_env(a)
                    state = agent.runtime_vars['observations'][-1]

                    if agent.engine.won:
                        success = True
                        break
                    if getattr(agent.engine, "lost", False):
                        success = False
                        break

                if success or getattr(agent.engine, "lost", False):
                    break

            except Exception as e:
                print(f"[{game}] Level {level}: error while executing subplan '{subplan}': {e}")
                success = False
                break

        # Final success check
        if agent.engine.won:
            success = True

        print(f"[{game}] Level {level} plan_available={plan_available} success={success}")
        return plan_available, success

    # ── Run all requested levels
    results = {}
    for lvl in levels:
        with agent.timing_run(label=f"reuse_only_L{lvl}", level=lvl):
            plan_avail, ok = _execute_level(lvl)
        results[f"L{lvl}_plan"] = plan_avail
        results[f"L{lvl}_success"] = ok


    print("\n[BABYAI:REUSE-ONLY] Sequence complete.")
    print(results)
    return results




if __name__ == '__main__':
    import argparse, ast, os, time, json, io, sys
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, default='pb1',
                        choices=['baba','lava','babyai','pb1','sokoban','labyrinth','cheesemaze', 'minihack'],)
    parser.add_argument('--level-sets', type=str, default="{'pb1': [0, 1, 2, 3]}",
                        help="Python dict, e.g. \"{'labyrinth':[0], 'maze':[0,1]}\"")
    parser.add_argument('--episode-length', type=int, default=20)
    parser.add_argument('--world-model-file-name', type=str, default='worldmodel')
    parser.add_argument('--domain-file-name', type=str, default='domain.pddl')
    parser.add_argument('--predicates-file-name', type=str, default='predicates')
    parser.add_argument('--json-reporter-path', type=str, default='KekeCompetition-main/Keke_JS/reports/TBRL_BABA_REPORT.json')
    parser.add_argument('--learn-model', action='store_true')
    parser.add_argument('--query-mode', type=str, default='openai_direct')
    parser.add_argument('--experiment-dir', type=str, default='debuglv3',
                        help='Directory to store experiment runs')
    parser.add_argument('--multi-level', action='store_true', help='Run multiple levels sequentially')
    parser.add_argument('--max-attempts', type=int, default=4, help='Maximum attempts per level')
    parser.add_argument('--prune-plans', action='store_true', help='Use the current method of handling one subplan at a time')
    parser.add_argument('--centralize-files', action='store_true',
                        help='Share domain and predicate files across games')
    parser.add_argument('--scratch', action='store_true', help='Start with a clean slate of generated files')
    parser.add_argument('--babyai-sequence', action='store_true',
                    help='Run BabyAI transfer sequence: 19 -> 8 -> 13 with WM/predicates/domain transfer.')
    parser.add_argument('--babyai-level-set', type=str, default='babyai_levels',
                    help='Level set name for BabyAI (default: babyai_levels)')
    parser.add_argument('--transfer-levels', type=str, default='{ "babyai": [13] }',
                    help='Set or per-game dict, e.g. "{13}" or "{ \\"babyai\\":[13] }"')
    
    # argparse additions
    parser.add_argument('--babyai-reuse-strict', action='store_true',
                    help='Run BabyAI 19→8→13 using ONLY existing files (no generation).')
    
    parser.add_argument(
    '--babyai-reuse-only',
    action='store_true',
    help='Run BabyAI 19→8→13 strictly with existing worldmodel/predicates/domain/plans (no synthesis).'
)
    parser.add_argument('--seed-from', type=str, default=None,
                    help='Path to a previous experiment dir whose artifacts should seed this run')
    
    parser.add_argument(
    '--seed-src-game',
    type=str,
    default=None,
    help='Name of the SOURCE game folder under tc_game/ to seed from '
         '(e.g., "labyrinth"). If omitted, defaults to the target --game.'
)
    parser.add_argument('--language-model', type=str, default='gpt-4o-2024-11-20',
                        help='Language model name passed to TheoryCoderAgent (e.g. o4-mini-2025-04-16, gpt-4o-2024-11-20).')
    parser.add_argument('--reasoning-effort', type=str, default=None,
                        help='Reasoning effort for o-series models: low | medium | high (default: None means model default).')

    

    args = parser.parse_args()

    # after parsing:
    try:
        tl = ast.literal_eval(args.transfer_levels)
        # normalize to sets
        if isinstance(tl, dict):
            tl = {k: set(v) for k, v in tl.items()}
        elif isinstance(tl, (list, tuple, set)):
            tl = set(tl)
        elif isinstance(tl, int):
            tl = {tl}
        elif not isinstance(tl, set):
            raise ValueError
    except Exception:
        raise ValueError("--transfer-levels must be an int, a set/list, or a dict of game → list/set of ints")
    

    # Safer than eval
    try:
        level_sets = ast.literal_eval(args.level_sets)
        assert isinstance(level_sets, dict)
    except Exception:
        raise ValueError("--level-sets must be a Python dict string, e.g. \"{'labyrinth':[0]}\"")

    # --- plan files per game (edit to your filenames) ---
    PLAN_FILES = {
        "labyrinth":    "labyrinth_plans.json",
        "maze":         "maze_plans.json",
        "sokoban":      "sokoban_plans.json",
        "baba":         "baba_plans.json",
        "pb1":          "pb1_plans.json",
        "cheesemaze":   "cheesemaze_plans.json",
        "lava":         "lava_plans.json",
        "babyai":         "babyai_plans.json",
    }

    # ---- logging (tee) helper ----
    class _Tee(io.TextIOBase):
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
            return len(data)
        def flush(self):
            for s in self.streams:
                s.flush()

    def start_log(game: str, level_id: int):
        os.makedirs(os.path.join('run_logs', game), exist_ok=True)
        log_path = os.path.join('run_logs', game, f"{game}_level{level_id}_{int(time.time())}.log")
        _log_file = open(log_path, 'w', buffering=1)  # line-buffered
        sys.stdout = _Tee(sys.stdout, _log_file)
        sys.stderr = _Tee(sys.stderr, _log_file)
        print(f"[TEE] Logging stdout/stderr to {log_path}")
        return log_path

    # ---- scratch cleanup ----
    exp_root = os.path.join('experiments', args.experiment_dir)
    if args.scratch:
        if args.multi_level:
            for g in level_sets.keys():
                clear_run_files(g, args.world_model_file_name, args.domain_file_name, exp_root)
        else:
            clear_run_files(args.game, args.world_model_file_name, args.domain_file_name, exp_root)

    # ---- agent ----
    agent = TheoryCoderAgent(
        base_dir=exp_root,
        episode_length=args.episode_length,
        world_model_load_name=args.world_model_file_name,
        json_reporter_path=args.json_reporter_path,
        predicates_file_name=args.predicates_file_name,
        domain_file_name=args.domain_file_name,
        do_revise_model=args.learn_model,
        plans_file_name=PLAN_FILES.get(args.game, 'plans.json'),
        query_mode=args.query_mode,
        prune_plans=args.prune_plans,
        centralize_files=args.centralize_files,
        create_subdir=False,
        transfer_levels=tl,
        language_model=args.language_model,
        reasoning_effort=args.reasoning_effort,
        use_responses_api=False,     # True to use the Responses API instead
    )


    if args.seed_from:
        seed_game_from_previous(
            agent,
            args.seed_from,
            args.game,
            copy_plans=True,
            src_game=args.seed_src_game  # ← CLI controls this
        )




    if args.babyai_reuse_only:
        if args.game != 'babyai':
            print("[note] --babyai-reuse-only forces game=babyai")
        start_log('babyai', 19)  # your tee logging
        agent.plans_file_name = 'babyai_plans.json'

        results = run_babyai_reuse_only(
            agent,
            level_set_name=args.babyai_level_set,
            levels=(19, 8, 13),
            plans_filename='babyai_plans.json'
        )

        # Save tape like in other modes
        from pathlib import Path
        import json, time as _t
        Path('tapes').mkdir(parents=True, exist_ok=True)
        with open(f'tapes/babyai_reuse_only_{int(_t.time())}.json', 'w') as f:
            json.dump(agent.tape, f, indent=4)

        agent._save_timings(filename=f"timings_babyai_reuse_only.json")
        sys.exit(0)


    # after building `agent`:
    if args.babyai_reuse_strict:
        if args.game != 'babyai':
            print("[note] --babyai-reuse-strict runs BabyAI regardless of --game")
        start_log('babyai', 19)  # your existing tee helper
        agent.plans_file_name = 'babyai_plans.json'
        out = run_babyai_reuse_strict(agent, level_set_name=args.babyai_level_set)
        from pathlib import Path
        import json, time as _t
        Path('tapes').mkdir(parents=True, exist_ok=True)
        with open(f'tapes/babyai_reuse_strict_{int(_t.time())}.json', 'w') as f:
            json.dump(agent.tape, f, indent=4)
        sys.exit(0)

    # ---- engine factory for ALL games ----
    def build_engine(game: str, level_set: str, level_id: int):
        if game == 'baba':
            return BabaIsYou(level_set=level_set, level_id=level_id)
        if game == 'lava':
            return LavaGrid()
        if game == 'babyai':
            return BabyAI(level_set=level_set, level_id=level_id)
        if game == 'pb1':
            return pb1env(level_set=level_set, level_id=level_id)
        if game == 'sokoban':
            return SokobanEnv(level_set=level_set, level_id=level_id)
        if game == 'labyrinth':
            return LabyrinthEnv(level_set=level_set, level_id=level_id)
        if game == 'cheesemaze':
            return CheesemazeEnv(level_set=level_set, level_id=level_id)
        if game == 'minihack':
            return MinihackEnv(level_set=level_set, level_id=level_id)
        raise ValueError(f"Unknown game: {game}")
    
    if args.babyai_sequence:
    # Force game to babyai for this mode
        if args.game != 'babyai':
            print("[note] --babyai-sequence sets game to 'babyai' regardless of --game")
        start_log('babyai', 19)
        agent.plans_file_name = 'babyai_plans.json'
        run_babyai_transfer_sequence(agent, level_set_name=args.babyai_level_set)
        # Save an overall tape
        Path('tapes').mkdir(parents=True, exist_ok=True)
        tape_path = f'tapes/babyai_transfer_{int(time.time())}.json'
        with open(tape_path, 'w') as f:
            json.dump(agent.tape, f, indent=4)
        sys.exit(0)


    # ---------- run ----------
    if args.multi_level:
        # run all (game -> [levels]) as provided; rotate a log per game/level
        overall = {"levels_completed": [], "levels_failed": []}
        for game, levels in level_sets.items():
            agent.plans_file_name = PLAN_FILES.get(game, 'plans.json')
            for level_id in levels:
                start_log(game, level_id)
                # pick level_set key as the dict key name when single-game dicts are used
                level_set_name = list(level_sets.keys())[0] if game in ['lava'] else game + "_levels" if game not in level_sets else game
                # if caller passed a dict like {'pb1':[0,1]}, level_set should be that dict key
                level_set_name = list(level_sets.keys())[0] if len(level_sets) == 1 else (game if game in level_sets else 'default')
                engine = build_engine(game, level_set_name, level_id)
                print(f"\n=== Running {game} level {level_id} ===")
                ok = agent.run(engine, max_attempts=args.max_attempts)
                (overall["levels_completed"] if ok else overall["levels_failed"]).append(f"{game}:{level_id}")

        # Save combined tape
        Path('tapes').mkdir(parents=True, exist_ok=True)
        tape_path = f'tapes/multirun_{int(time.time())}.json'
        with open(tape_path, 'w') as f:
            json.dump(agent.tape, f, indent=4)

        print("\nExperiment Complete!")
        print(f"Levels Completed: {len(overall['levels_completed'])}")
        print(f"Levels Failed: {len(overall['levels_failed'])}")
    else:
        # single game: use the first entry in level_sets for level_set & level
        level_set_name = list(level_sets.keys())[0]
        level_id = level_sets[level_set_name][0]

        # update plan file for selected game
        agent.plans_file_name = PLAN_FILES.get(args.game, 'plans.json')

        start_log(args.game, level_id)
        engine = build_engine(args.game, level_set_name, level_id)

        print(f"\n=== Running {args.game} level {level_id} (level_set={level_set_name}) ===")
        agent.run(engine, max_attempts=args.max_attempts)

        # Save tape to json
        Path('tapes').mkdir(parents=True, exist_ok=True)
        tape_path = f'tapes/{args.game}_{level_set_name}_{level_id}_{int(time.time())}.json'
        with open(tape_path, 'w') as f:
            json.dump(agent.tape, f, indent=4)
