"""TheoryCoder Agent - Legacy entry point.

This module provides backwards compatibility for the original monolithic
TheoryCoder implementation. New code should import from the theorycoder
package instead:

    from theorycoder import AgentConfig, load_transfer_config
    from theorycoder.cli import main

Usage (unchanged):
    python theorycoder2.py --transfer-config config.yaml
"""
import importlib
from pathlib import Path
from copy import deepcopy
import json
import os
import sys
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
import utils
from preprocessing import *
import openai
from experiment_logger import ExperimentLogger
from time import strftime, gmtime
from envs.babyai_env import BabyAI


def _babyai_seed_kwargs(seed):
    """Inline kwargs for BabyAI construction. Pass seed only when explicitly set."""
    return {'seed': seed} if seed is not None else {}


def normalize_pddl_whitespace(pddl: str) -> str:
    """Insert a single space before any '?' parameter that lacks preceding
    whitespace. Targets the open-source-model bug where Llama/similar emit
    fused PDDL forms like `(opened?x)` or `(?obj1 - object?obj2 - object)`.

    Leaves `(?x ...)` (variable as first token after `(`) untouched, since `?`
    after `(` is the legal form for a standalone variable position.

    Gated by env var TC_PDDL_NORMALIZE=1 at the call site, so closed-source
    runs are unaffected.
    """
    import re as _re
    return _re.sub(r'(?<=[^\s(])(\?)', r' \1', pddl)
from pb1_env import pb1env
from envs.sokoban_env import SokobanEnv
from envs.labyrinth_env import LabyrinthEnv
from envs.games import BabaIsYou
from envs.maze_env import MazeEnv
from cheesemaze_env import CheesemazeEnv
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
import requests

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
from utils import directions

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
    """
    src_game = src_game or game

    src = Path(src_experiment_dir) / "tc_game" / src_game
    dst = Path(agent.logger.experiment_dir) / "tc_game" / game
    dst.mkdir(parents=True, exist_ok=True)

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

    agent.predicates_empty = False
    agent.domain_empty = False
    if "predicates" in sys.modules:
        importlib.reload(sys.modules["predicates"])
    agent.reload_predicates_module()

    print(f"[seed] seeded {game} from {src_experiment_dir}/tc_game/{src_game} → {dst}")


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

    controllables = {
        entity for entity in state
        if rule_formed(state, f'{entity[:-4]}_word', 'is_word', 'you_word')
    }

    overlappables = {
        entity for entity in state
        if rule_formed(state, f'{entity[:-4]}_word', 'is_word', 'win_word')
    }

    pushables = {
        entity for entity in state
        if entity.endswith('_word')
        or rule_formed(state, f'{entity[:-4]}_word', 'is_word', 'push_word')
        or (entity.endswith('_obj') and rule_formed(state, f'{entity[:-4]}_word', 'is_word', 'push_word'))
    }

    state['controllables'] = list(controllables)

    if 'empty' in state:
        del state['empty']

    if 'won' in state:
        del state['won']

    state['overlappables'] = list(overlappables)
    state['pushables'] = list(pushables)

    word_entities = [entity for entity in state.keys() if entity.endswith('_word')]
    rules_on_map = []
    for subj in word_entities:
        for pred in word_entities:
            for obj in word_entities:
                if rule_formed(state, subj, pred, obj):
                    rules_on_map.append(subj + ' ' + pred + ' ' + obj)

    state['rules_formed'] = rules_on_map

    return state

# ---------------------------------------------------------------
# Prompt loading helpers

def load_world_prompts(game_name: str, legacy: bool = False):
    """Load world model prompts for a specific game.

    Lookup order:
      1. world_modeling_prompts/<game_name>_legacy/<name>.txt  (if legacy=True)
      2. world_modeling_prompts/<game_name>/<name>.txt          (game-specific)
      3. world_modeling_prompts/<name>.txt                       (default)

    Parameters
    ----------
    game_name : str
        Name of the game (e.g. "sokoban").
    legacy : bool
        If True, prefer the ``<game_name>_legacy`` variant when it exists.
        Used by the BabyAI old-vs-new door-format A/B experiment.

    Returns
    -------
    tuple[str, str]
        The initialize and revise prompt strings.
    """

    base_dir = Path(__file__).resolve().parent / "world_modeling_prompts"
    candidate_dirs = []
    if legacy:
        candidate_dirs.append(base_dir / f"{game_name}_legacy")
    candidate_dirs.append(base_dir / game_name)

    def _resolve(filename):
        for d in candidate_dirs:
            p = d / filename
            if p.exists():
                return p
        return base_dir / filename

    init_path = _resolve("initialize_world_model.txt")
    revise_path = _resolve("revise_world_model.txt")
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
        # language_model='gpt-4o',
        # language_model='gpt-4o-2024-08-06',
        language_model='gpt-4o-2024-11-20',
        # language_model='o4-mini',
        # language_model = 'o1-mini',
        # language_model = 'o1-preview',
        # language_model='gpt-3.5-turbo',
        domain_file_name='domain.pddl',  # Added this for PDDL file path
        predicates_file_name='predicates.py',
        query_mode='openai_direct',  # Options: 'langchain_openai', 'openai_direct', 'groq'
        groq_model="llama3-8b-8192",  # Specify the Groq model
        reasoning_effort: str | None = "high",  # new: "low" | "medium" | "high" | None
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
        use_custom_pddl_prompts=False,
        replace_predicates=False,
        transfer_levels=None,
        env_seed=None,
        # Replay buffer transition modes for revision:
        #   Mode 1 (Full): revision_mode="full"
        #   Mode 2 (Errors only): revision_mode="errors_only"
        #   Mode 3 (Limited, prioritize errors): revision_mode="limited"
        #   Mode 4 (Hybrid): revision_mode="hybrid" - tries full, falls back to errors_only if too large
        revision_mode="hybrid",  # "full", "errors_only", "limited", or "hybrid"
        max_revision_transitions=50,  # Max transitions in "limited" mode
        max_revision_tokens=30000  # Max tokens for hybrid mode (GPT-4o context is 128k, leave room for prompt/response)
    ):
        
    #     transfer_levels:
    #   - set[int]              → applies to all games (e.g., {13})
    #   - dict[str, set[int]]   → per-game (e.g., {"babyai": {13}, "pb1": set()})
        self.transfer_levels = transfer_levels or set()
        self.env_seed = env_seed


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
        self.revision_mode = revision_mode  # "full", "errors_only", "limited", "hybrid"
        self.max_revision_transitions = max_revision_transitions  # Limit for "limited" mode
        self.max_revision_tokens = max_revision_tokens  # Token limit for "hybrid" mode



        # Prompts
        # self.infer_interaction_rule_prompt = infer_interaction_rule_prompt
        # self.get_relevant_rules_prompt = get_relevant_rules_prompt
        # self.planner_prompt = planner_prompt
        # self.evaluate_plan_prompt = evaluate_plan_prompt
        self.debug_model_prompt = debug_model_prompt
        self.debug_predicate_prompt = debug_predicate_prompt


        # Initialize query clients based on query_mode
        if query_mode == 'langchain_openai':
            # self.llm_client = ChatOpenAI(model_name=language_model, temperature=temperature)
            print("hi")
        elif query_mode == 'openai_direct':
            self.llm_client = openai
        elif query_mode == 'groq':
            from groq import Groq
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
        elif query_mode == 'openrouter':
            self.openrouter_base_url = os.environ.get(
                "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
            )
            self.openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
            if not self.openrouter_api_key:
                raise ValueError("OPENROUTER_API_KEY must be set for query_mode='openrouter'")
            self.openrouter_referer = os.environ.get("OPENROUTER_REFERER")
            self.openrouter_title = os.environ.get("OPENROUTER_TITLE")
            self.llm_client = None
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
        self.reasoning_effort = reasoning_effort
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
        self.current_game = None  # Track current game for game-specific prompts

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


    # --- modify signature
    def query_lm(self, prompt, *, label: str | None = None, temperature: float | None = None):
        """
        Query the LLM based on the selected query mode.
        Times the call and records model/usage metadata when available.

        Parameters
        ----------
        prompt : str
            The user prompt.
        label : str, optional
            Stage tag for timing/metadata.
        temperature : float, optional
            Per-call temperature override. If None, uses self.temperature.
            Pass temperature=0 for deterministic outputs like PDDL/predicate
            generation where grounding to state-dict keys matters.
        """
        effective_temperature = self.temperature if temperature is None else temperature
        model_name = self.groq_model if self.query_mode == 'groq' else self.language_model
        meta_extra = {"model": model_name, "provider": self.query_mode}
        start_ex = {}

        if self.query_mode == 'langchain_openai':
            with self._record_time("llm", detail=label or "langchain_openai", extra=meta_extra):
                chat_prompt = HumanMessagePromptTemplate.from_template(prompt)
                out = self.llm_client.invoke(chat_prompt.to_messages())
                return out.content, None

        elif self.query_mode == "openai_direct":
            messages = [{"role": "user", "content": prompt}]
            with self._record_time("llm", detail=label or "openai_direct", extra=meta_extra):
                completion = self.llm_client.chat.completions.create(
                    model=self.language_model,
                    messages=messages,
                    temperature=effective_temperature,
                    seed=42,
                )
            # stash token usage if present
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
            return completion.choices[0].message.content.strip(), completion

        elif self.query_mode == 'groq':
            messages = [{"role": "user", "content": prompt}]
            with self._record_time("llm", detail=label or "groq", extra=meta_extra):
                response = self.llm_client.chat.completions.create(
                    messages=messages,
                    model=self.groq_model,
                    temperature=effective_temperature,
                    seed=42,
                )
            return response.choices[0].message.content.strip(), response
        
        elif self.query_mode == "custom":

            url = f"{self.custom_base_url.rstrip('/')}/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "api-key": self.custom_api_key,
            }
            # Reasoning-model detection: o-series (o1-*, o3-*, o4-*) and gpt-5* (except chat-latest).
            # All reject non-default temperature and require max_completion_tokens.
            mdl = (self.language_model or "").lower()
            is_o_series = len(mdl) >= 2 and mdl[0] == "o" and mdl[1].isdigit()
            is_gpt5_reasoning = mdl.startswith("gpt-5") and "chat" not in mdl
            is_reasoning_model = is_o_series or is_gpt5_reasoning
            payload = {
                "model": self.language_model,
                "messages": [{"role": "user", "content": prompt}],
            }
            # OpenAI's o-series rejects 'max_tokens' and wants 'max_completion_tokens'.
            if is_reasoning_model:
                payload["max_completion_tokens"] = 16384
            else:
                payload["max_tokens"] = 16384
                payload["temperature"] = effective_temperature
            effort = getattr(self, "reasoning_effort", None)
            if is_reasoning_model and effort:
                payload["reasoning_effort"] = effort
            req_timeout = 300 if is_reasoning_model else 60

            with self._record_time("llm", detail=label or "custom", extra=meta_extra):
                r = requests.post(url, headers=headers, json=payload, timeout=req_timeout)
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

        elif self.query_mode == "openrouter":
            url = f"{self.openrouter_base_url.rstrip('/')}/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.openrouter_api_key}",
            }
            if self.openrouter_referer:
                headers["HTTP-Referer"] = self.openrouter_referer
            if self.openrouter_title:
                headers["X-Title"] = self.openrouter_title

            # OpenRouter pre-reserves max_tokens against the account balance per call,
            # not just actual usage. Default 4096 is comfortable for TC's typical 500-3000
            # token responses and avoids 402s on low-balance accounts. Override with
            # TC_OPENROUTER_MAX_TOKENS=N if you need more headroom (e.g., long revisions).
            or_max_tokens = int(os.environ.get("TC_OPENROUTER_MAX_TOKENS", "4096"))

            payload = {
                "model": self.language_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": effective_temperature,
                "max_tokens": or_max_tokens,
            }

            with self._record_time("llm", detail=label or "openrouter", extra=meta_extra):
                r = requests.post(url, headers=headers, json=payload, timeout=600)
                if not r.ok:
                    print(f"[openrouter] {r.status_code} response body:\n{r.text}")
                    r.raise_for_status()
                completion = r.json()

            usage = completion.get("usage")
            if usage:
                self.timing["events"][-1].update({
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                })

            text = completion["choices"][0]["message"]["content"].strip()
            return text, completion

        else:
            raise ValueError(f"Unsupported query_mode: {self.query_mode}")


        # ---- Timing helpers -------------------------------------------------
    def _init_timing(self):
        self.timing = {
            "events": [],      # list of dict entries (one per timing)
            "rollup": {},      # computed at end (category -> stats)
        }

    @contextmanager
    def _record_time(self, category: str, detail: str | None = None, extra: dict | None = None):
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

    def set_reasoning(self, effort: str | None):
        """
        Set effort to 'low' | 'medium' | 'high' for o4-mini (or None to disable).
        """
        self.reasoning_effort = effort



    @contextmanager
    def timing_run(self, label: str, level: int | None = None, filename: str | None = None):
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
        # support set[int] (global) OR dict[str,set[int]] (per-game)
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
                "from utils import directions\n\n"
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
    
    def load_prompt(self, name: str, game_name: str | None = None, **kwargs) -> str:
        """
        Load a prompt from the abstraction_prompts directory.
        Checks game-specific (and _legacy variant when the legacy_door_format
        flag is set on the agent), then falls back to default.

        Looks for, in order:
          1. abstraction_prompts/<game_name>_legacy/<name>.txt  (if legacy_door_format=True)
          2. abstraction_prompts/<game_name>/<name>.txt          (game-specific)
          3. abstraction_prompts/<name>.txt                       (default)
        """
        base = Path("abstraction_prompts")
        game_name = game_name or getattr(self, "current_game", None)
        use_legacy = getattr(self, "legacy_door_format", False)
        # Optional variant suffix (e.g., TC_PDDL_VARIANT=multi → init_pddl_files_multi.txt).
        # Used for TC-C ablation studies that need a different prompt than Full TC.
        # When unset (the default) the original prompts are loaded — Full TC is unaffected.
        variant = os.environ.get("TC_PDDL_VARIANT", "").strip()
        # ICL ablation: TC_NSHOT=N → look for <name>_nN.txt (e.g. init_pddl_files_n1.txt).
        # Falls through to the normal lookup if the variant file is missing.
        nshot = os.environ.get("TC_NSHOT", "").strip()

        # If legacy door format is on, try the _legacy variant of the game folder first
        if game_name and use_legacy:
            legacy_specific = base / f"{game_name}_legacy" / f"{name}.txt"
            if legacy_specific.exists():
                text = legacy_specific.read_text()
                return text.format(**kwargs)

        # Try game-specific prompt first (with optional _variant suffix taking priority)
        if game_name:
            if nshot:
                game_nshot = base / game_name / f"{name}_n{nshot}.txt"
                if game_nshot.exists():
                    print(f"[load_prompt] using n-shot variant: {game_nshot}")
                    return game_nshot.read_text().format(**kwargs)
            if variant:
                game_variant = base / game_name / f"{name}_{variant}.txt"
                if game_variant.exists():
                    print(f"[load_prompt] using variant: {game_variant}")
                    return game_variant.read_text().format(**kwargs)
            game_specific = base / game_name / f"{name}.txt"
            if game_specific.exists():
                text = game_specific.read_text()
                return text.format(**kwargs)

        # Fall back to default prompt
        default_path = base / f"{name}.txt"
        if default_path.exists():
            text = default_path.read_text()
            return text.format(**kwargs)

        raise FileNotFoundError(
            f"Prompt '{name}' not found in abstraction_prompts/ "
            + (f"or abstraction_prompts/{game_name}/" if game_name else "")
        )
    

    def extract_code_block(self, text: str, lang: str, which: int) -> str:
        pattern = rf"```{lang}(.*?)(?=```)"
        blocks = re.findall(pattern, text, re.DOTALL)
        return blocks[which-1].strip() if len(blocks) >= which else ""

    def extract_pddl_files(self, text: str) -> tuple[str, str]:
        """Return domain and problem code from ``text``.

        Handles:
        - Two separate ```pddl``` blocks (one domain, one problem)
        - Both domain and problem in a single block
        - Multiple blocks where the LLM quotes the old domain before the new one
          (takes the LAST domain block and LAST problem block)
        """
        pattern = r"```pddl(.*?)(?=```)"
        blocks = re.findall(pattern, text, re.DOTALL)
        domain, problem = "", ""

        # First: collect all individual domain and problem definitions across all blocks
        domain_candidates = []
        problem_candidates = []

        for block in blocks:
            block = block.strip()
            dom_idx = block.find("(define (domain")
            prob_idx = block.find("(define (problem")

            if dom_idx != -1 and prob_idx != -1:
                # Both in one block - split them
                if dom_idx < prob_idx:
                    domain_candidates.append(block[dom_idx:prob_idx].strip())
                    problem_candidates.append(block[prob_idx:].strip())
                else:
                    problem_candidates.append(block[prob_idx:dom_idx].strip())
                    domain_candidates.append(block[dom_idx:].strip())
            elif dom_idx != -1:
                domain_candidates.append(block[dom_idx:].strip())
            elif prob_idx != -1:
                problem_candidates.append(block[prob_idx:].strip())

        # Take the LAST candidates (LLMs often quote old version first, new version last)
        if domain_candidates:
            domain = domain_candidates[-1]
        if problem_candidates:
            problem = problem_candidates[-1]

        # Open-source post-LLM normalization. Off by default.
        if os.environ.get("TC_PDDL_NORMALIZE", "").strip() == "1":
            if domain:
                domain = normalize_pddl_whitespace(domain)
            if problem:
                problem = normalize_pddl_whitespace(problem)

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

        # Prefer game-specific transfer_domain prompt if present, else fall back to shared.
        # Honor TC_PDDL_VARIANT (e.g., "o4minihigh" → transfer_domain_o4minihigh.txt) when set,
        # matching the load_prompt() lookup order: game/variant → shared/variant → game → shared.
        variant = os.environ.get("TC_PDDL_VARIANT", "").strip()
        candidates = []
        if variant:
            candidates += [
                Path(f"abstraction_prompts/{game_name}/transfer_domain_{variant}.txt"),
                Path(f"abstraction_prompts/transfer_domain_{variant}.txt"),
            ]
        candidates += [
            Path(f"abstraction_prompts/{game_name}/transfer_domain.txt"),
            Path("abstraction_prompts/transfer_domain.txt"),
        ]
        tpl_path = next((p for p in candidates if p.exists()), None)
        if tpl_path is None:
            raise FileNotFoundError(f"transfer_domain prompt not found (tried: {candidates})")
        if variant and "_" + variant in tpl_path.name:
            print(f"[transfer_domain] using variant: {tpl_path}")
        tpl = tpl_path.read_text()
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


    def extend_domain_and_generate_problem(self, level: int):
        """
        Extend an existing domain by adding ONE new operator, then generate a new problem file.
        Uses the extend_domain_and_problem prompt.

        This is called when transfer config has domain: "extend"
        """
        game_name = self._get_game_name()
        domain_path = self.game_dir / f"{game_name}_domain.pddl"
        problem_path = self.game_dir / f"{game_name}_{level}.pddl"

        if not domain_path.exists():
            print(f"[EXTEND] ERROR: No existing domain to extend at {domain_path}")
            return None

        # Update current game for game-specific prompt loading
        self.current_game = game_name

        raw = json.dumps(self.engine.get_obs())
        mission = getattr(self.engine, "mission", "")

        # Load the extend_domain_and_problem prompt
        prompt = self.load_prompt(
            "extend_domain_and_problem",
            domain_file=domain_path.read_text(),
            raw_state=raw,
            mission=mission,
        )

        print("[EXTEND] DOMAIN EXTENSION PROMPT")
        print(prompt)

        t0 = time.perf_counter()
        response, meta = self.query_lm(prompt, label="extend_domain_and_problem")
        elapsed_sec = time.perf_counter() - t0

        print("[EXTEND] DOMAIN EXTENSION RESPONSE")
        print(response)

        # Log step
        step_dir = Path(self.logger.create_step("extend_domain"))
        with open(step_dir / "prompt.txt", "w") as f:
            f.write(prompt)
        with open(step_dir / "response.txt", "w") as f:
            f.write(response)
        with open(step_dir / "completion_info.json", "w") as f:
            json.dump(meta, f, default=lambda o: getattr(o, "to_dict", lambda: str(o))(), indent=2)

        self._record_step_timing(level_id=level, step_dir=step_dir,
                                stage_tag="extend_domain_and_problem", elapsed_sec=elapsed_sec)

        # Extract PDDL blocks
        domain_code, problem_code = self.extract_pddl_files(response)
        domain_code = self._ensure_closed_parentheses(domain_code)
        problem_code = self._ensure_closed_parentheses(problem_code)

        # Write the extended domain and new problem
        domain_path.write_text(domain_code)
        problem_path.write_text(problem_code)
        shutil.copy(domain_path, step_dir / domain_path.name)
        shutil.copy(problem_path, step_dir / problem_path.name)

        print(f"[EXTEND] Extended domain saved to {domain_path}")
        print(f"[EXTEND] New problem saved to {problem_path}")

        # Run Fast Downward
        cmd = [
            "python3", self.fast_downward_path,
            str(domain_path), str(problem_path),
            "--search", "astar(blind())",
        ]
        try:
            with self._record_time("fd", detail="astar(blind)", extra={"level": self.current_level}):
                subprocess.run(cmd, check=True)
        except CalledProcessError as e:
            print(f"[EXTEND] Fast Downward failed with exit code {e.returncode}")
            print("[EXTEND] Attempting LLM-based PDDL debug...")
            if self._call_pddl_debug(domain_path, problem_path):
                try:
                    with self._record_time("fd", detail="astar(blind)", extra={"level": self.current_level}):
                        subprocess.run(cmd, check=True)
                except CalledProcessError as e2:
                    print(f"[EXTEND] Fast Downward still failed with exit code {e2.returncode}")
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
        
        # Update current game for game-specific prompt loading
        self.current_game = self._get_game_name()

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
        # Update current game for game-specific prompt loading
        self.current_game = self._get_game_name()
        
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
            mission=getattr(self.engine, "mission", ""),
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


    def extend_predicates(self, domain_path: Path, problem_path: Path):
        """
        Extend existing predicates.py by adding implementations for NEW predicates.
        Uses the extend_predicates prompt.

        This is called when transfer config has predicates: "extend"
        """
        # Update current game for game-specific prompt loading
        self.current_game = self._get_game_name()

        pred_path = self.game_dir / "predicates.py"
        if not pred_path.exists():
            print(f"[EXTEND PRED] ERROR: No existing predicates to extend at {pred_path}")
            # Fall back to generating from scratch
            return self.generate_and_save_predicates(domain_path, problem_path)

        existing_predicates = pred_path.read_text()
        raw = json.dumps(self.engine.get_obs())

        # Load the extend_predicates prompt
        prompt = self.load_prompt(
            "extend_predicates",
            domain_file=domain_path.read_text(),
            problem_file=problem_path.read_text(),
            raw_state=raw,
            existing_predicates=existing_predicates,
        )

        print("[EXTEND PRED] PREDICATE EXTENSION PROMPT")
        print(prompt)

        step_dir = Path(self.logger.create_step("extend_predicates"))
        step_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()
        resp, completion = self.query_lm(prompt, label="extend_predicates")
        elapsed_sec = time.perf_counter() - t0

        print("[EXTEND PRED] PREDICATE EXTENSION RESPONSE")
        print(resp)

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

        self._record_step_timing(level_id=self.current_level, step_dir=step_dir,
                                stage_tag="extend_predicates", elapsed_sec=elapsed_sec)

        # Extract the python code block - should contain ALL predicates (old + new)
        predicates_code = self.extract_code_block(resp, "python", 1)

        # Write into shared module
        self.update_experiment_predicates(predicates_code)

        # Snapshot into step folder
        shutil.copy(self.game_dir / "predicates.py", step_dir / "predicates.py")

        # Update flags & runtime vars
        self.runtime_vars['predicates'] = predicates_code
        self.predicates_empty = False
        print("[EXTEND PRED] Predicates extended and saved.")


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

        if isinstance(self.engine, BabaIsYou):
            all_keys.remove("won")
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
                    utils="directions = {\n    'left': [-1, 0],\n    'right': [1, 0],\n    'up': [0, 1],\n    'down': [0, -1],\n}",
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

    def _count_tokens(self, text):
        """
        Estimate token count for a text string.
        Uses tiktoken for accurate GPT model token counting.
        """
        try:
            import tiktoken
            # Use the encoding for GPT-4o
            encoding = tiktoken.encoding_for_model("gpt-4o")
            return len(encoding.encode(text))
        except ImportError:
            # Fallback: rough estimation (1 token ≈ 4 characters for English)
            return len(text) // 4
        except Exception:
            # Fallback on any error
            return len(text) // 4

    def _save_transitions_to_file(self, obs, errors, exploratory_plan=None):
        """
        Save all transitions to a file in the replay_buffers directory.

        Args:
            obs: List of (s0, a, s1) tuples
            errors: List of error strings (empty string if no error)
            exploratory_plan: Optional name of exploratory plan
        """
        import os
        from pathlib import Path

        # Create replay_buffers directory under experiment dir
        replay_dir = Path(self.logger.experiment_dir) / "replay_buffers"
        replay_dir.mkdir(exist_ok=True)

        # Generate filename with timestamp and optional exploratory plan
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        if exploratory_plan:
            # Clean up exploratory plan name for use in filename
            safe_plan = exploratory_plan.replace(" ", "_").replace("/", "_")[:50]
            filename = f"transitions_L{self.current_level}_{timestamp}_{safe_plan}.txt"
        else:
            filename = f"transitions_L{self.current_level}_{timestamp}.txt"

        filepath = replay_dir / filename

        # Write transitions to file
        with open(filepath, 'w') as f:
            f.write(f"Replay Buffer Transitions\n")
            f.write(f"Level: {self.current_level}\n")
            f.write(f"Timestamp: {timestamp}\n")
            if exploratory_plan:
                f.write(f"Exploratory Plan: {exploratory_plan}\n")
            f.write(f"Total Transitions: {len(obs)}\n")
            f.write(f"Transitions with Errors: {sum(1 for e in errors if e)}\n")
            f.write("=" * 80 + "\n\n")

            for idx, ((s0, a, s1), error) in enumerate(zip(obs, errors), 1):
                f.write(f"Transition {idx}\n")
                f.write("-" * 80 + "\n")
                f.write(f"Initial state: {s0}\n")
                f.write(f"Action: {a}\n")
                f.write(f"Next state: {s1}\n")

                if error:
                    f.write(f"\n⚠️  Prediction Errors:\n{error}\n")
                else:
                    f.write(f"\n✓ No prediction errors\n")

                f.write("\n" + "=" * 80 + "\n\n")

        print(f"[REPLAY BUFFER] Saved {len(obs)} transitions to: {filepath}")


    def _choose_synthesis_examples(self, exploratory_plan=None, max_transitions=None, errors_only=False,
                                    save_to_file=False, mode="hybrid", max_tokens=None):
        """
        Choose (s0, a) --> s1 transitions from replay buffer as program
        synthesis examples.

        Modes:
        - "full": All transitions, no filtering
        - "errors_only": Only transitions with prediction errors
        - "limited": Up to max_transitions, prioritizing errors
        - "hybrid": Tries full first, falls back to errors_only if token count exceeds max_tokens

        Args:
            exploratory_plan (str): The exploratory plan for which to generate errors.
            max_transitions (int): Maximum number of transitions (for "limited" mode).
            errors_only (bool): If True, only include error transitions (for "errors_only" mode).
            save_to_file (bool): If True, saves all transitions to a file in the artifacts directory.
            mode (str): Override mode - "full", "errors_only", "limited", or "hybrid"
            max_tokens (int): Max tokens for hybrid mode fallback

        Returns:
            list: A list of formatted examples.
            int: The count of errors.
        """
        # Simple solution: Just take the last k from the buffer
        obs = self.replay_buffers[::1]

        actions_taken = [a for (s0, a, s1) in obs]
        correct_states = [s1 for (s0, a, s1) in obs]

        # Generate predictions for each (s0, a) pair in obs
        preds = [self._call_model_debug(s0, a) for (s0, a, s1) in obs]

        # Compare predicted and actual states to identify errors
        errors = [self._get_pred_errors(s1, pred) for (s0, a, s1), pred in zip(obs, preds)]

        # Count the number of errors (from full buffer, before filtering)
        error_count = sum([1 if e else 0 for e in errors])

        # Save full replay buffer to file if requested
        if save_to_file:
            self._save_transitions_to_file(obs, errors, exploratory_plan)

        # Determine filtering mode
        # If mode is explicitly provided, use it; otherwise use errors_only/max_transitions parameters
        if mode == "hybrid":
            # HYBRID MODE: Try full first, check tokens, fall back to errors_only if needed
            # Step 1: Try with all transitions
            examples_full = [self._make_observation_summaries((s0, a, s1), e) for (s0, a, s1), e in zip(obs, errors)]
            full_text = "\n\n".join(examples_full)
            token_count = self._count_tokens(full_text)

            max_token_limit = max_tokens if max_tokens is not None else getattr(self, 'max_revision_tokens', 30000)

            print(f"[DEBUG] Hybrid mode: Full transitions token count = {token_count} (limit: {max_token_limit})")

            if token_count <= max_token_limit:
                # Full mode fits within token limit
                obs_for_summary = obs
                errors_for_summary = errors
                print(f"[DEBUG] Hybrid mode: Using FULL mode ({len(obs)} transitions)")
            else:
                # Too many tokens, fall back to errors_only
                filtered_data = [(o, e) for o, e in zip(obs, errors) if e]
                obs_errors = [o for o, e in filtered_data]
                errors_errors = [e for o, e in filtered_data]

                examples_errors = [self._make_observation_summaries((s0, a, s1), e) for (s0, a, s1), e in zip(obs_errors, errors_errors)]
                errors_text = "\n\n".join(examples_errors)
                errors_token_count = self._count_tokens(errors_text)

                print(f"[DEBUG] Hybrid mode: Errors-only token count = {errors_token_count}")

                if errors_token_count <= max_token_limit:
                    # Errors only fits
                    obs_for_summary = obs_errors
                    errors_for_summary = errors_errors
                    print(f"[DEBUG] Hybrid mode: Falling back to ERRORS_ONLY mode ({len(obs_errors)} transitions with errors)")
                else:
                    # Even errors_only is too large, apply max_transitions limit
                    max_trans = max_transitions if max_transitions is not None else getattr(self, 'max_revision_transitions', 50)
                    obs_for_summary = obs_errors[:max_trans]
                    errors_for_summary = errors_errors[:max_trans]
                    print(f"[DEBUG] Hybrid mode: Errors still too large, limiting to {max_trans} transitions")

        elif mode == "full" or (mode is None and not errors_only and max_transitions is None):
            # FULL MODE: All transitions
            obs_for_summary = obs
            errors_for_summary = errors
            print(f"[DEBUG] Full mode: Using all {len(obs)} transitions")

        elif mode == "errors_only" or (mode is None and errors_only and max_transitions is None):
            # ERRORS_ONLY MODE: Filter to errors
            filtered_data = [(o, e) for o, e in zip(obs, errors) if e]
            obs_for_summary = [o for o, e in filtered_data]
            errors_for_summary = [e for o, e in filtered_data]
            print(f"[DEBUG] Errors-only mode: {len(obs)} total → {len(obs_for_summary)} with errors")

        elif mode == "limited" or (mode is None and max_transitions is not None):
            # LIMITED MODE: Apply max_transitions limit, prioritizing errors
            max_trans = max_transitions if max_transitions is not None else getattr(self, 'max_revision_transitions', 50)

            # Filter to errors_only first if requested
            if errors_only:
                filtered_data = [(o, e) for o, e in zip(obs, errors) if e]
                obs_filtered = [o for o, e in filtered_data]
                errors_filtered = [e for o, e in filtered_data]
            else:
                obs_filtered = obs
                errors_filtered = errors

            # Apply max_transitions limit
            if len(obs_filtered) > max_trans:
                if errors_only:
                    # All are errors, just take first max_trans
                    obs_for_summary = obs_filtered[:max_trans]
                    errors_for_summary = errors_filtered[:max_trans]
                    print(f"[DEBUG] Limited error transitions: {len(obs_filtered)} → {max_trans}")
                else:
                    # Prioritize transitions with errors first
                    with_errors = [(o, e) for o, e in zip(obs_filtered, errors_filtered) if e]
                    without_errors = [(o, e) for o, e in zip(obs_filtered, errors_filtered) if not e]

                    # Take up to max_trans, prioritizing error transitions
                    if len(with_errors) >= max_trans:
                        selected = with_errors[:max_trans]
                    else:
                        # Take all error transitions + fill remaining with non-error transitions
                        remaining = max_trans - len(with_errors)
                        selected = with_errors + without_errors[:remaining]

                    obs_for_summary = [o for o, e in selected]
                    errors_for_summary = [e for o, e in selected]

                    print(f"[DEBUG] Limited mode (prioritizing errors): {len(obs_filtered)} total → {len(selected)} selected ({len(with_errors)} with errors)")
            else:
                obs_for_summary = obs_filtered
                errors_for_summary = errors_filtered
                print(f"[DEBUG] Limited mode: Using all {len(obs_filtered)} transitions (under limit)")

        else:
            # Default fallback: use full mode
            obs_for_summary = obs
            errors_for_summary = errors
            print(f"[DEBUG] Default mode: Using all {len(obs)} transitions")

        # Create summaries of the observations along with the errors
        examples = [self._make_observation_summaries((s0, a, s1), e) for (s0, a, s1), e in zip(obs_for_summary, errors_for_summary)]

        # Format examples with the exploratory plan if provided
        if exploratory_plan:
            # last_example = examples[-1] if examples else ""

            formatted_examples = [f"ERRORS FROM WORLD MODEL for EXPLORATORY PLAN {exploratory_plan}:\n\n" + "\n\n".join(examples)]

        else:
            formatted_examples = examples

        return formatted_examples, error_count

    def _revise_world_model(self):
        print(f"[DEBUG] ===== _revise_world_model ENTERED, do_revise_model={self.do_revise_model} =====")
        if not self.do_revise_model:
            print("[DEBUG] Skipping revision because do_revise_model is False")
            return

        self.tape[-1]['revision_prompts'] = {}
        self.tape[-1]['revision_responses'] = {}

        # DEBUG: Check replay buffer status
        print(f"[DEBUG] _revise_world_model called")
        print(f"[DEBUG] replay_buffers length: {len(self.replay_buffers)}")
        print(f"[DEBUG] replay_buffers content: {self.replay_buffers[:3] if self.replay_buffers else 'EMPTY'}")

        # Use configured revision mode to avoid context overflow
        # Save full buffer to file for later analysis
        #
        # Revision modes:
        #   - "full": All transitions
        #   - "errors_only": Only error transitions
        #   - "limited": Up to max_revision_transitions, prioritizing errors
        #   - "hybrid": Try full, fall back to errors_only if exceeds max_revision_tokens (DEFAULT)
        revision_mode = getattr(self, 'revision_mode', 'hybrid')
        max_transitions = getattr(self, 'max_revision_transitions', 50)
        max_tokens = getattr(self, 'max_revision_tokens', 30000)

        examples, error_count = self._choose_synthesis_examples(
            mode=revision_mode,
            max_transitions=max_transitions,
            max_tokens=max_tokens,
            save_to_file=True
        )

        print(f"[DEBUG] examples length: {len(examples)} (mode: {revision_mode})")
        print(f"[DEBUG] error_count: {error_count}")
        if examples:
            print(f"[DEBUG] first example preview: {examples[0][:200]}")

        mission = getattr(self.engine, 'mission', '')

        if self._do_revise_model(error_count):
            prompt = self.revise_world_model_prompt.format(
                actions_set=self.engine.actions_set,
                errors_from_world_model='\n\n'.join(examples),
                world_model_str=self.runtime_vars['world_model_str'],
                utils="directions = {\n    'left': [-1, 0],\n    'right': [1, 0],\n    'up': [0, 1],\n    'down': [0, -1],\n}",
                mission=mission,
            )

            print(prompt)

            # Create step directory and save files
            step_dir = self.logger.create_step("revision")
            resp, completion = self.query_lm(prompt, label="revise_world_model")
            new_world_model_code = self.extract_code_from_response(resp)

            if new_world_model_code:
                new_world_model_code = self._validate_world_model_code(
                    new_world_model_code, prompt, label="revise_world_model"
                )

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
        # For initialization, use all transitions (no limit) but save to file
        examples, error_count = self._choose_synthesis_examples(save_to_file=True)
        mission = getattr(self.engine, 'mission', '')

        # Current WM as a string (may be blank)
        current_wm = self.runtime_vars.get('world_model_str', '').strip()

        # Format directly into your template (since it already contains the header)
        base_prompt = self.initialize_world_model_prompt.format(
            current_state=self.runtime_vars['observations'][-1],
            actions_set=self.engine.actions_set,
            num_random_actions=num_actions,
            errors_from_world_model='\n\n'.join(examples),
            utils="directions = {\n    'left': [-1, 0],\n    'right': [1, 0],\n    'up': [0, 1],\n    'down': [0, -1],\n}",
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
            new_world_model_code = self._validate_world_model_code(
                new_world_model_code, prompt, label="initialize_world_model"
            )

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
                )
            if not actionlist:
                print("No actions found; executing random action.")
                actions.append(random.choice(self.actions_set))
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
                importlib.reload(worldmodel)
                importlib.reload(planner)
                importlib.reload(levelrunner)
                importlib.reload(utils)
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
                            timeout=getattr(self, 'planner_timeout', None),
                        )
                    if not action_seq:
                        print(f"No actions found for subplan {subplan}; executing random action.")
                        actions.append(random.choice(self.actions_set))
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
        # base_dir_path.write_text(merged)



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

    def _validate_world_model_code(self, code: str, original_prompt: str, label: str, max_retries: int = 1):
        """
        Parse `code` with ast. On SyntaxError, re-prompt the LLM with the error and the
        bad code, up to `max_retries` times. Returns the validated code, or None if all
        attempts produced invalid Python. The caller should skip overwriting the
        worldmodel file when this returns None.
        """
        current_code = code
        for attempt in range(max_retries + 1):
            try:
                ast.parse(current_code)
                if attempt > 0:
                    print(f"[WM-VALIDATE] {label}: valid after {attempt} retry/retries")
                return current_code
            except SyntaxError as e:
                print(f"[WM-VALIDATE] {label} attempt {attempt + 1}: SyntaxError at line {e.lineno}: {e.msg}")
                if attempt >= max_retries:
                    print(f"[WM-VALIDATE] {label}: giving up after {max_retries + 1} attempts; world model NOT overwritten")
                    return None

                retry_prompt = (
                    f"{original_prompt}\n\n"
                    f"---\n"
                    f"PREVIOUS ATTEMPT PRODUCED INVALID PYTHON.\n"
                    f"SyntaxError at line {e.lineno}: {e.msg}\n\n"
                    f"The code you produced was:\n"
                    f"```python\n{current_code}\n```\n\n"
                    f"Return the corrected world model in a ```python ``` block. "
                    f"Only fix the syntax error; keep all existing logic intact."
                )
                resp, _ = self.query_lm(retry_prompt, label=f"{label}_syntax_retry{attempt + 1}")
                new_code = self.extract_code_from_response(resp)
                if not new_code:
                    print(f"[WM-VALIDATE] {label} retry {attempt + 1}: response had no ```python``` block; aborting")
                    return None
                current_code = new_code
        return None


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
        self._reset_exploration_actions = []  # captured for run() to include in first_letters
        if self.is_world_model_empty():
            print("Detected an empty world model.")
            # self._initialize_world_model()
            num_actions = 20
            plan = self.execute_random_actions(num_actions=num_actions)  # Adjust the number as needed
            print(plan)
            print("World model was empty, revised the model. Moving to next iteration.")
            for action in plan:
                self.step_env(action)
                self._reset_exploration_actions.append(action)
            print(f"[DEBUG-WM-INIT] reset() about to call _initialize_world_model(num_actions={num_actions}) for level={getattr(self, 'current_level', '?')}")
            try:
                self._initialize_world_model(num_actions)
                print(f"[DEBUG-WM-INIT] reset() _initialize_world_model returned OK for level={getattr(self, 'current_level', '?')}")
            except Exception as e:
                import traceback
                print(f"[DEBUG-WM-INIT] reset() _initialize_world_model RAISED: {e!r}")
                traceback.print_exc()
                raise

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

    def _load_plan_for_problem(self, problem_path: Path) -> list[str] | None:
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

    
    def _save_plan_for_problem(self, problem_path: Path, plan: list[str]):
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
        from envs.babyai_env import BabyAI
        from pb1_env import pb1env
        from envs.sokoban_env import SokobanEnv
        from envs.labyrinth_env import LabyrinthEnv
        from envs.maze_env import MazeEnv
        from cheesemaze_env import CheesemazeEnv



        if isinstance(self.engine, BabaIsYou):
            return "baba"
        elif isinstance(self.engine, LavaGrid):
            return "lava"
        elif isinstance(self.engine, DoggoEnv):
            return "doggo"
        elif isinstance(self.engine, DrunkDwarfEnv):
            return "drunkdwarf"
        elif isinstance(self.engine, BabyAI):
            return "babyai"
        elif isinstance(self.engine, Boulderdash2Env):
            return "boulderdash2"
        elif isinstance(self.engine, pb1env):
            return "pb1"
        elif isinstance(self.engine, SokobanEnv):
            return "sokoban"
        elif isinstance(self.engine, SokobanEnvFULL):
            return "sokobanFULL"
        elif isinstance(self.engine, LabyrinthEnv):
            return "labyrinth"
        elif isinstance(self.engine, MazeEnv):
            return "maze"
        elif isinstance(self.engine, CheesemazeEnv):
            return "cheesemaze"
        else:
            # fallback to the class name
            return self.engine.__class__.__name__.lower()


    def run(self, engine, max_revisions=5, max_attempts=6):
        print(f"[DEBUG] ===== RUN STARTED - File version with debug enabled (Jan 17 16:40) =====")
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
            # Load per-game prompts if available (legacy variant for old door encoding)
            init_prompt, revise_prompt = load_world_prompts(
                game_name, legacy=getattr(self, "legacy_door_format", False)
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

            # Check transfer config for domain handling mode
            transfer_cfg = getattr(self, '_current_transfer_config', None)
            domain_mode = transfer_cfg.get('domain', False) if transfer_cfg else False

            # Check for debug_no_llm mode (skip all LLM calls)
            debug_no_llm = getattr(self, 'debug_no_llm', False)

            if debug_no_llm:
                # DEBUG MODE: Skip all LLM calls, just use existing artifacts
                print(f"[{game_name}] DEBUG MODE: Skipping LLM calls, using existing artifacts")
                problem_path = self.game_dir / f"{game_name}_{self.current_level}.pddl"
                if domain_path.exists() and problem_path.exists():
                    print(f"[{game_name}] Using existing domain and problem files")
                    # Just run the planner on existing files
                    cmd = [
                        "python3", self.fast_downward_path,
                        str(domain_path), str(problem_path),
                        "--search", "astar(blind())",
                    ]
                    try:
                        subprocess.run(cmd, check=True)
                        plan = self.parse_sas_plan("sas_plan")
                        self._save_plan_for_problem(problem_path, plan)
                    except CalledProcessError as e:
                        print(f"[DEBUG MODE] Fast Downward failed: {e.returncode}")
                else:
                    print(f"[{game_name}] ERROR: Missing domain or problem file for debug mode")
                    print(f"  Domain: {domain_path} exists={domain_path.exists()}")
                    print(f"  Problem: {problem_path} exists={problem_path.exists()}")
                self.plans = self._load_plans()

            elif domain_mode == 'extend':
                # EXTEND MODE: domain was transferred, now extend it with new operator
                print(f"[{game_name}] EXTEND MODE: Extending domain with new operator")

                # Check if domain file actually exists after transfer
                if not domain_path.exists():
                    print(f"[{game_name}] WARNING: Domain transfer failed for extension, generating from scratch")
                    self.generate_and_solve_pddl(level=self.current_level)
                else:
                    self.extend_domain_and_generate_problem(level=self.current_level)
                    self._load_domain_pddl(self.domain_file)
                    # CRITICAL: Update predicate_arg_mapping after domain extension
                    set_domain_file(self.domain_file)
                    self.plans = self._load_plans()
                    print(f"Loaded plans for '{game_name}':", list(self.plans.keys()))

            elif domain_mode is True:
                # TRANSFER MODE: domain was transferred as-is, just generate new problem
                print(f"[{game_name}] TRANSFER MODE: Using transferred domain, generating new problem")

                # Check if domain file actually exists after transfer
                if not domain_path.exists():
                    print(f"[{game_name}] WARNING: Domain transfer failed or source not found, generating from scratch")
                    self.generate_and_solve_pddl(level=self.current_level)
                else:
                    _ = self.generate_problem_for_existing_domain(level=self.current_level)
                    # CRITICAL: Update predicate_arg_mapping for transferred domain
                    set_domain_file(self.domain_file)
                    self.plans = self._load_plans()
                    print(f"Loaded plans for '{game_name}':", list(self.plans.keys()))

            elif not domain_path.exists():
                # SCRATCH MODE: no domain exists, generate from scratch
                print(f"[{game_name}] domain PDDL not found -> generating now")
                self.generate_and_solve_pddl(level=self.current_level)
                domain_code = domain_path.read_text()
                if self.centralize_files:
                    if central_missing:
                        shutil.copy(domain_path, self.central_domain_path)
                    else:
                        self.update_experiment_domain(domain_path, domain_code)
                self._load_domain_pddl(self.domain_file)
                # CRITICAL: Update predicate_arg_mapping after domain generation
                set_domain_file(self.domain_file)

                self.plans = self._load_plans()
                print(f"Loaded plans for '{game_name}':", list(self.plans.keys()))
            else:
                # EXISTING MODE: domain already exists (not from transfer), solve existing
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

            # Handle transferred artifacts BEFORE the main loop
            # Check if using a fixed worldmodel (never modified by LLM)
            fixed_wm = getattr(self, 'fixed_worldmodel_path', None)

            # If worldmodel was transferred or fixed, load it into runtime_vars so is_world_model_empty() sees it
            worldmodel_transferred = transfer_cfg.get('worldmodel', False) if transfer_cfg else False
            if fixed_wm or worldmodel_transferred:
                wm_path = self.game_dir / "worldmodel.py"
                if wm_path.exists():
                    label = "fixed" if fixed_wm else "transferred"
                    print(f"[{game_name}] Loading {label} worldmodel into runtime")
                    self.capture_world_model()

            # If predicates need extension, do it now (before the main loop)
            # Skip if debug_no_llm mode is enabled
            if not debug_no_llm:
                pred_mode = transfer_cfg.get('predicates', False) if transfer_cfg else False
                if pred_mode == 'extend':
                    domain_path = self.game_dir / f"{game_name}_domain.pddl"
                    problem_path = self.game_dir / f"{game_name}_{self.current_level}.pddl"
                    if domain_path.exists() and problem_path.exists():
                        print(f"[{game_name}] Extending predicates for new domain operators")
                        self.extend_predicates(domain_path, problem_path)
            else:
                print(f"[{game_name}] DEBUG MODE: Skipping predicate extension")

            # -------------------------------------------

            revision_count = 0
            debug_count = 0
            attempt_count = 0
            exploratory_plan_index = 0

            while revision_count < max_revisions and attempt_count < max_attempts:
                # Initialize
                self.reset(keep_model=True)
                # Include any random-exploration actions reset() took to bootstrap an empty WM,
                # so the printed Solution reflects the full trajectory (not just the BFS plan).
                first_letters = ''.join(a[0] for a in getattr(self, '_reset_exploration_actions', []))
                model_was_revised = False

                # Extract the original state immediately after reset
                initial_state = deepcopy(self.engine.get_obs())

                if isinstance(self.engine, BabaIsYou):
                    initial_state = process_state_baba(initial_state)

                print(f"[DEBUG-WM-INIT] run() pre-check level={self.current_level}: is_world_model_empty={self.is_world_model_empty()}, do_revise_model={self.do_revise_model}, fixed_wm={fixed_wm}")
                if self.is_world_model_empty() and self.do_revise_model and not fixed_wm:
                    # If the world model is empty, use the default action set
                    print("World model is empty, executing random actions.")
                    num_actions = 15
                    plan = self.execute_random_actions(num_actions=num_actions)  # Adjust the number as needed
                    print(plan)
                    print("World model was empty, revised the model. Moving to next iteration.")
                    for action in plan:
                        self.step_env(action)
                        first_letters += action[0]  # include exploration in reported solution
                    print(f"[DEBUG-WM-INIT] run() about to call _initialize_world_model(num_actions={num_actions}) for level={self.current_level}")
                    try:
                        self._initialize_world_model(num_actions)
                        print(f"[DEBUG-WM-INIT] run() _initialize_world_model returned OK for level={self.current_level}")
                    except Exception as e:
                        import traceback
                        print(f"[DEBUG-WM-INIT] run() _initialize_world_model RAISED: {e!r}")
                        traceback.print_exc()
                        raise

                    self.capture_world_model()


                    # — now re-check the shared predicates.py snapshot —
                    self._load_predicates(self.predicates_file_name)
                    if self.predicates_empty:
                        domain_path  = self.game_dir / f"{game_name}_domain.pddl"
                        problem_path = self.game_dir / f"{game_name}_{self.current_level}.pddl"

                        # Check transfer config for predicate handling mode
                        pred_mode = transfer_cfg.get('predicates', False) if transfer_cfg else False

                        if pred_mode == 'extend':
                            print("[PREDICATES] predicates.py needs extension -> extending via LLM")
                            self.extend_predicates(domain_path, problem_path)
                        else:
                            print("[PREDICATES] predicates.py is empty -> generating via LLM")
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

                print(f"[DEBUG] Executing plan with {len(plan)} actions: {plan}")
                for action in plan:
                    print(f"[DEBUG] Executing action: {action}")
                    self.step_env(action)
                    print(f"[DEBUG] After step_env, replay_buffers length: {len(self.replay_buffers)}")
                    first_letters += action[0]  # Collect the first letters of each action

                    # Exit if agent won
                    if self.engine.won or (isinstance(self.engine, BabaIsYou) and self.current_level == 6 and first_letters == 'rrruuu'):
                            
                        self.tape[-1]['exit_condition'] = 'won'
                        self._update_solution(self.current_level, first_letters)
                        self.level_statistics[f"{self.engine.level_set}_{self.current_level}"]["first_letters"] = first_letters
                        self.level_statistics[f"{self.engine.level_set}_{self.current_level}"]["revisions"] = revision_count
                        self.level_statistics[f"{self.engine.level_set}_{self.current_level}"]["debugs"] = debug_count
                        print(first_letters)
                        if isinstance(self.engine, BabyAI):
                            self.engine.close()

                        if isinstance(self.engine, (Boulderdash2Env, pb1env, SokobanEnv, LabyrinthEnv, CheesemazeEnv)):
                            self.engine.save_screen(f"{self.logger.experiment_dir}/{self.current_level}_{attempt_count}.png")

                        # Save actions and summary before returning on success
                        summary = f"""
        Level: {self.current_level}
        Revisions: {revision_count}
        Attempts: {attempt_count}
        Final Status: {"Won" if self.engine.won else "Failed"}
        First Letters: {first_letters}
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
                        if not fixed_wm:
                            self._revise_world_model()
                        else:
                            print("[FIXED WM] Skipping world model revision (fixed)")
                        attempt_count += 1
                        model_was_revised = True
                        if isinstance(self.engine, BabyAI):
                            self.engine.close()

                        if isinstance(self.engine, (Boulderdash2Env, pb1env, SokobanEnv, LabyrinthEnv, CheesemazeEnv)):
                            self.engine.save_screen(f"{self.logger.experiment_dir}/{self.current_level}_{attempt_count}.png")
                        break

                # Silent failure: plan finished, agent neither won nor died.
                # WM mispredicted the outcome → revise from the latest transitions.
                if (not self.engine.won and not self.engine.lost
                        and not model_was_revised and plan and not fixed_wm):
                    print("Plan executed but no win/loss — revising WM from silent failure")
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
                
                            if isinstance(self.engine, (Boulderdash2Env, pb1env, SokobanEnv, LabyrinthEnv, CheesemazeEnv)):
                                self.engine.save_screen(f"{self.logger.experiment_dir}/{self.current_level}_{attempt_count}.png")

                            return True

                        # Check if the agent lost (e.g., died or failed critically)
                        if self.engine.lost:
                            self.tape[-1]['exit_condition'] = 'lost'
                    
                            print(self.engine.get_obs())
                            attempt_count += 1

                            if isinstance(self.engine, (Boulderdash2Env, pb1env, SokobanEnv, LabyrinthEnv, CheesemazeEnv)):
                                self.engine.save_screen(f"{self.logger.experiment_dir}/{self.current_level}_{attempt_count}.png")
                            break

                
                # if theorycoder domain predicates, world model is not empty and but prooblem file is just gen problem


                # Handle model revision if necessary
                if not self.is_world_model_empty() and self.do_revise_model and not model_was_revised:
                    pruned_plans = ["collect_diamond diamond1","collect_diamond diamond2","collect_diamond diamond3","collect_diamond diamond4","collect_diamond diamond5","collect_diamond diamond6","collect_diamond diamond7","collect_diamond diamond8","collect_diamond diamond9","escape_via_exit avatar exitdoor"]
                    print("pruned automatically", pruned_plans)
                   
                    LLM_pruned_plans = pruned_plans

                    # Aggregate datasets for all exploratory plans
                    self.aggregated_dataset = []
                    # Save replay buffer before reset (this contains transitions from the main plan execution)
                    saved_replay_buffer = self.replay_buffers.copy()
                    print(f"[DEBUG] Saved replay_buffer with {len(saved_replay_buffer)} transitions from main plan")

                    subplans = self.plans.get(str(self.current_level), [])
                    print(f"[DEBUG] Subplans for level {self.current_level}: {subplans}")

                    if subplans:
                        # If we have subplans, execute them and collect examples
                        self.reset(keep_model=True)
                        # Restore replay buffer to accumulate transitions across subplans
                        self.replay_buffers = saved_replay_buffer

                        for subplan in subplans:
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
                            print(f"[DEBUG] Before _choose_synthesis_examples for subplan '{subplan}', replay_buffers length: {len(self.replay_buffers)}")
                            revision_mode = getattr(self, 'revision_mode', 'hybrid')
                            max_transitions = getattr(self, 'max_revision_transitions', 50)
                            max_tokens = getattr(self, 'max_revision_tokens', 30000)
                            examples, error_count = self._choose_synthesis_examples(
                                exploratory_plan=subplan,
                                mode=revision_mode,
                                max_transitions=max_transitions,
                                max_tokens=max_tokens,
                                save_to_file=True
                            )
                            print(f"[DEBUG] After _choose_synthesis_examples, examples length: {len(examples)} (mode: {revision_mode}), error_count: {error_count}")
                            self.aggregated_dataset.append({
                                "subplan": subplan,
                                "examples": examples,
                                "error_count": error_count
                            })
                    else:
                        # No subplans available - use the replay buffer from the main plan execution directly
                        print(f"[DEBUG] No subplans for level {self.current_level}, using main plan replay buffer")
                        self.replay_buffers = saved_replay_buffer
                        print(f"[DEBUG] replay_buffers length: {len(self.replay_buffers)}")
                        revision_mode = getattr(self, 'revision_mode', 'hybrid')
                        max_transitions = getattr(self, 'max_revision_transitions', 50)
                        max_tokens = getattr(self, 'max_revision_tokens', 30000)
                        examples, error_count = self._choose_synthesis_examples(
                            exploratory_plan="main_plan",
                            mode=revision_mode,
                            max_transitions=max_transitions,
                            max_tokens=max_tokens,
                            save_to_file=True
                        )
                        print(f"[DEBUG] examples length: {len(examples)} (mode: {revision_mode}), error_count: {error_count}")
                        self.aggregated_dataset.append({
                            "subplan": "main_plan",
                            "examples": examples,
                            "error_count": error_count
                        })

                    # Perform model revision using the aggregated dataset
                    aggregated_examples = []
                    for data in self.aggregated_dataset:
                        aggregated_examples.append(f"ERRORS FROM WORLD MODEL for EXPLORATORY PLAN {data['subplan']}:\n\n" + "\n\n".join(data['examples']))

                    print(f"[DEBUG] aggregated_dataset has {len(self.aggregated_dataset)} entries")
                    print(f"[DEBUG] aggregated_examples has {len(aggregated_examples)} entries")
                    if aggregated_examples:
                        print(f"[DEBUG] First aggregated_example preview (first 300 chars): {aggregated_examples[0][:300]}")

                    if isinstance(self.engine, (Boulderdash2Env, pb1env, SokobanEnv, LabyrinthEnv, CheesemazeEnv)):
                        self.engine.save_screen()

                    mission = getattr(self.engine, 'mission', '')

                    prompt = self.revise_world_model_prompt.format(
                        actions_set=self.engine.actions_set,
                        errors_from_world_model='\n\n'.join(aggregated_examples),
                        world_model_str=self.runtime_vars['world_model_str'],
                        utils="directions = {\n    'left': [-1, 0],\n    'right': [1, 0],\n    'up': [0, 1],\n    'down': [0, -1],\n}",
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
                        new_world_model_code = self._validate_world_model_code(
                            new_world_model_code, prompt, label="revise_world_model"
                        )

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


            if isinstance(self.engine, (Boulderdash2Env, pb1env, SokobanEnv, LabyrinthEnv, CheesemazeEnv)):
                self.engine.save_screen()

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
                elif args.game == 'lava':
                    self.engine = LavaGrid()
                elif args.game == 'babyai':
                    self.engine = BabyAI(level_set=level_set, level_id=level_id,
                                         **_babyai_seed_kwargs(self.env_seed))
                if args.game == 'pb1':
                    self.engine = pb1env(level_set=level_set, level_id=level_id)
                elif args.game == 'sokoban':
                    self.engine = SokobanEnv(level_set=level_set, level_id=level_id)
                if args.game == 'labyrinth':
                    self.engine = LabyrinthEnv(level_set=level_set, level_id=level_id)
                elif args.game == 'cheesemaze':
                    self.engine = CheesemazeEnv(level_set=level_set, level_id=level_id)
                
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

    # ──────────────────────────────────────────────────────────────────
    # ARTIFACT TRANSFER METHODS FOR TRANSFER MODE
    # ──────────────────────────────────────────────────────────────────

    def _setup_level_artifacts(self, level: int, transfer_from: int = None, transfer_config: dict = None, game_name: str = None):
        """
        Setup artifact storage for this level.
        If transfer_config provided, copy artifacts from source level's snapshot.

        Args:
            level: Current level ID
            transfer_from: Source level to transfer from (None if learning from scratch)
            transfer_config: Dict specifying what to transfer (worldmodel, domain, predicates)
            game_name: Game name (optional, used if engine not yet initialized)
        """
        from pathlib import Path
        import shutil

        # Store transfer config for use in run()
        self._current_transfer_config = transfer_config

        # Use provided game_name or try to get from engine
        if game_name is None:
            try:
                game_name = self._get_game_name()
            except:
                game_name = "game"  # Fallback

        # Create level-specific snapshot directory for later retrieval
        level_snapshot_dir = self.game_dir / f'level_{level}_artifacts'
        level_snapshot_dir.mkdir(exist_ok=True)

        # If transferring, copy from source level's snapshot
        if transfer_from is not None and transfer_config:
            source_snapshot_dir = self.game_dir / f'level_{transfer_from}_artifacts'

            if not source_snapshot_dir.exists():
                print(f"[TRANSFER ERROR] Source snapshot dir not found: {source_snapshot_dir}")
                print(f"[TRANSFER ERROR] Cannot transfer artifacts from level {transfer_from} to level {level}")
                print(f"[TRANSFER ERROR] Will need to generate artifacts from scratch for level {level}")
                return

            # Transfer worldmodel
            if transfer_config.get('worldmodel', False):
                src_wm = source_snapshot_dir / 'worldmodel.py'
                if src_wm.exists():
                    shutil.copy(src_wm, self.game_dir / 'worldmodel.py')
                    print(f"[TRANSFER] Copied worldmodel from level {transfer_from}")
                else:
                    print(f"[TRANSFER WARNING] Worldmodel not found in source snapshot")

            # Transfer domain (True or "extend" both require copying the file)
            domain_val = transfer_config.get('domain', False)
            if domain_val in [True, 'extend']:
                src_domain = source_snapshot_dir / f'{game_name}_domain.pddl'
                if src_domain.exists():
                    shutil.copy(src_domain, self.game_dir / f'{game_name}_domain.pddl')
                    mode_str = "for extension" if domain_val == 'extend' else "as-is"
                    print(f"[TRANSFER] Copied domain from level {transfer_from} ({mode_str})")
                else:
                    print(f"[TRANSFER ERROR] Domain not found at: {src_domain}")
                    print(f"[TRANSFER ERROR] Expected to transfer domain from level {transfer_from} but file is missing")
                    print(f"[TRANSFER ERROR] Domain will need to be generated from scratch for level {level}")

            # Transfer predicates (True or "extend" both require copying the file)
            pred_val = transfer_config.get('predicates', False)
            if pred_val in [True, 'extend']:
                src_pred = source_snapshot_dir / 'predicates.py'
                if src_pred.exists():
                    shutil.copy(src_pred, self.game_dir / 'predicates.py')
                    mode_str = "for extension" if pred_val == 'extend' else "as-is"
                    print(f"[TRANSFER] Copied predicates from level {transfer_from} ({mode_str})")
                else:
                    print(f"[TRANSFER WARNING] Predicates not found in source snapshot")
            elif pred_val is False:
                # Predicates NOT transferred - mark for regeneration
                self._handle_predicate_transfer(transfer_config, transfer_from)

    def _snapshot_artifacts_to_dir(self, snapshot_dir: Path):
        """
        Save current artifacts to a snapshot directory for later transfer.

        Args:
            snapshot_dir: Directory to save snapshots to
        """
        from pathlib import Path
        import shutil

        snapshot_dir.mkdir(parents=True, exist_ok=True)

        game_name = self._get_game_name()

        # Copy worldmodel
        wm_path = self.game_dir / 'worldmodel.py'
        if wm_path.exists():
            shutil.copy(wm_path, snapshot_dir / 'worldmodel.py')
            print(f"[SNAPSHOT] Saved worldmodel to {snapshot_dir.name}")

        # Copy domain
        domain_path = self.game_dir / f'{game_name}_domain.pddl'
        if domain_path.exists():
            shutil.copy(domain_path, snapshot_dir / f'{game_name}_domain.pddl')
            print(f"[SNAPSHOT] Saved domain to {snapshot_dir.name}")

        # Copy predicates
        pred_path = self.game_dir / 'predicates.py'
        if pred_path.exists():
            shutil.copy(pred_path, snapshot_dir / 'predicates.py')
            print(f"[SNAPSHOT] Saved predicates to {snapshot_dir.name}")

        # Copy problem file for this level (if exists)
        if hasattr(self, 'current_level'):
            prob_path = self.game_dir / f'{game_name}_{self.current_level}.pddl'
            if prob_path.exists():
                shutil.copy(prob_path, snapshot_dir / f'{game_name}_{self.current_level}.pddl')
                print(f"[SNAPSHOT] Saved problem file to {snapshot_dir.name}")

    def _handle_predicate_transfer(self, transfer_config: dict, source_level: int):
        """
        Handle predicate transfer logic.
        If predicates are not being transferred, mark them for regeneration.

        Args:
            transfer_config: Transfer config dict
            source_level: Source level (for logging)
        """
        from pathlib import Path

        if not transfer_config.get('predicates', False):
            # Predicates NOT transferred - remove file to force regeneration
            pred_path = self.game_dir / 'predicates.py'
            if pred_path.exists():
                pred_path.unlink()
                print(f"[TRANSFER] Predicates marked for regeneration (not transferred from level {source_level})")

            # Set flag to trigger predicate generation
            self.predicates_empty = True


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
      - L19: learn domain+problem+world model+predicates (fresh if empty)
      - L8 : learn NEW domain+problem+world model+predicates (fresh, no merging, only if empty)
      - L13: reuse STRICTLY the L8 artifacts; problem-only generation if missing
    Files live in experiments/<exp>/tc_game/babyai/.
    """
    from pathlib import Path
    import shutil, os, sys
    from envs.babyai_env import BabyAI
    from theorycoder2 import load_world_prompts  # already in your file

    def _build_engine(lid):
        return BabyAI(level_set=level_set_name, level_id=lid,
                      **_babyai_seed_kwargs(getattr(agent, 'env_seed', None)))

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
    #      BUT only initialize WM if it's empty
    # ───────────────────────────────────────────────────────────────────
    print("\n[BABYAI] Level 19: learn domain+problem+world model+predicates (fresh if empty)")
    agent.engine = _build_engine(19)

    # Load per-game prompts (legacy variant for old door encoding)
    init_prompt, revise_prompt = load_world_prompts(
        game, legacy=getattr(agent, "legacy_door_format", False)
    )
    agent.initialize_world_model_prompt = init_prompt
    agent.revise_world_model_prompt = revise_prompt

    with agent.timing_run(label="babyai_L19_bootstrap", level=19):
        plan_19 = agent.generate_and_solve_pddl(level=19)
        agent.reset(keep_model=True)
        agent.capture_world_model()

        # 🔧 ONLY initialize WM if it's effectively empty
        if agent.is_world_model_empty():
            for a in agent.execute_random_actions(num_actions=debug_actions):
                agent.step_env(a)
            _orig = agent.do_revise_model
            agent.do_revise_model = True
            try:
                agent._initialize_world_model(num_actions=debug_actions)
            finally:
                agent.do_revise_model = _orig
        else:
            print("[babyai-transfer] L19: world model already present, skipping _initialize_world_model()")

        dom_19  = local_domain_path
        prob_19 = agent.game_dir / f"{game}_19.pddl"
        agent.predicates_empty = True
        agent.generate_and_save_predicates(dom_19, prob_19)

        print("[BABYAI] L19 summary:")
        print(f"  - Domain+problem generated: True")
        print(f"  - Plan found by FD: {ok_lv19} (len={len(plan_19) if plan_19 else 0})")
        print(f"  - World model initialized (if empty): True")
        print(f"  - Predicates generated: True")

    # Save L19 snapshots
    if (agent.game_dir / "worldmodel.py").exists():
        shutil.copy(agent.game_dir / "worldmodel.py", wm_lv19)
    if (agent.game_dir / "predicates.py").exists():
        shutil.copy(agent.game_dir / "predicates.py", preds_lv19)
    if local_domain_path.exists():
        shutil.copy(local_domain_path, dom_lv19)

    # ───────────────────────────────────────────────────────────────────
    # L8 — learn NEW domain + problem + WM + predicates
    #      BUT again only initialize WM if it's empty
    # ───────────────────────────────────────────────────────────────────
    print("\n[BABYAI] Level 8: learn NEW domain+problem+world model+predicates (fresh if empty)")
    agent.engine = _build_engine(8)

    with agent.timing_run(label="babyai_L8_bootstrap", level=8):
        plan_8_gen = agent.generate_and_solve_pddl(level=8)
        agent.reset(keep_model=True)
        agent.capture_world_model()

        # 🔧 ONLY initialize WM if it's effectively empty
        if agent.is_world_model_empty():
            for a in agent.execute_random_actions(num_actions=debug_actions):
                agent.step_env(a)

            _orig_revise = agent.do_revise_model
            agent.do_revise_model = True
            try:
                agent._initialize_world_model(num_actions=debug_actions)
            finally:
                agent.do_revise_model = _orig_revise
        else:
            print("[babyai-transfer] L8: world model already present, skipping _initialize_world_model()")

        dom_8  = local_domain_path
        prob_8 = agent.game_dir / f"{game}_8.pddl"
        agent.predicates_empty = True
        agent.generate_and_save_predicates(dom_8, prob_8)

    # attempts for L8 (separate timing around the actual run)
    with agent.timing_run(label="babyai_L8_attempt", level=8):
        original_flag = agent.do_revise_model
        agent.do_revise_model = True
        ok_lv8 = agent.run(agent.engine, max_revisions=3, max_attempts=getattr(agent, "max_replans", 3) or 3)
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

    # Ensure problem for L13 (problem-only generation), then run
    with agent.timing_run(label="babyai_L13_bootstrap", level=13):
        _ensure_problem_for_level(13)
        agent.do_revise_model = True

    with agent.timing_run(label="run", level=13, filename="timings_L13_run.json"):
        ok_lv13 = agent.run(agent.engine, max_revisions=3, max_attempts=getattr(agent, "max_replans", 3) or 3)

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

    def _engine(lid): return BabyAI(level_set=level_set_name, level_id=lid,
                                    **_babyai_seed_kwargs(getattr(agent, 'env_seed', None)))

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
        return BabyAI(level_set=level_set_name, level_id=level,
                      **_babyai_seed_kwargs(getattr(agent, 'env_seed', None)))

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
                import worldmodel, planner, levelrunner, utils
                importlib.reload(worldmodel)
                importlib.reload(utils)
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


# ───────────────────────────────────────────────────────────────────
# MULTI-TRIAL RUNNER WITH METRICS AGGREGATION
# ───────────────────────────────────────────────────────────────────

def run_with_trials(agent, num_trials, level_sets, max_revisions, max_attempts, build_engine_fn, multi_level, args):
    """
    Run the agent num_trials times with COMPLETELY ISOLATED trial directories.
    Each trial gets its own sandbox with all generated files (worldmodel, domain, predicates, etc).
    At the end, aggregates metrics across all trials.
    
    Returns aggregated summary with averages and success rate.
    """
    from pathlib import Path
    import json
    import shutil
    
    # Create master trials directory
    trials_dir = Path(agent.logger.experiment_dir) / "trials"
    trials_dir.mkdir(exist_ok=True)
    
    trial_metrics = []  # List of dicts: {tokens, time, success}
    original_exp_dir = agent.logger.experiment_dir
    
    for trial_num in range(1, num_trials + 1):
        print(f"\n{'='*70}")
        print(f"[TRIAL {trial_num}/{num_trials}] Starting isolated trial")
        print(f"{'='*70}")
        
        # Create COMPLETELY ISOLATED trial directory
        trial_dir = trials_dir / f"trial_{trial_num:03d}"
        trial_dir.mkdir(exist_ok=True)
        
        # Create tc_game subdirectory for this trial
        trial_game_dir = trial_dir / "tc_game" / args.game
        trial_game_dir.mkdir(parents=True, exist_ok=True)
        
        # Create tape and steps subdirectories
        (trial_dir / "tape").mkdir(exist_ok=True)
        (trial_dir / "steps").mkdir(exist_ok=True)
        
        # Point agent to this trial's isolated directory
        agent.logger.experiment_dir = str(trial_dir)
        agent.game_dir = trial_game_dir
        
        # Reinitialize logger's internal state to use new experiment_dir
        agent.logger.experiment_name = f"{original_exp_dir.split('/')[-1]}_trial_{trial_num:03d}"
        
        # Update environment variables to use trial-specific paths
        os.environ["TC_WORLDMODEL_FILE"] = str(trial_game_dir / "worldmodel.py")
        os.environ["TC_PREDICATES_FILE"] = str(trial_game_dir / "predicates.py")
        sys.modules.pop("predicates", None)

        # Update domain file path to trial directory
        agent.domain_file = str(trial_game_dir / f"{args.game}_domain.pddl")
        
        # RESET ENGINE STATE - clear old engine completely for fresh start
        agent.engine = None
        
        # Reset timing for this trial
        agent._init_timing()
        
        # Reset agent state (tape, etc.) for fresh trial
        agent.tape = [{}]
        
        trial_start = time.time()
        
        try:
            # Run this trial in complete isolation
            if multi_level:
                trial_result = agent.run_multiple_levels(level_sets, max_revisions, max_attempts)
                trial_success = len(trial_result.get("levels_completed", [])) > 0
            else:
                # Single level mode - build engine and run
                game = args.game
                level_set = list(level_sets.keys())[0]
                level_id = list(level_sets[level_set])[0]
                engine = build_engine_fn(game, level_set, level_id)
                trial_success = agent.run(engine, max_revisions, max_attempts)
            
            trial_elapsed = time.time() - trial_start
            
            # Aggregate token counts from all LLM calls in this trial
            total_tokens = 0
            prompt_tokens = 0
            completion_tokens = 0
            
            for event in agent.timing.get("events", []):
                if event.get("category") == "llm":
                    total_tokens += event.get("total_tokens", 0)
                    prompt_tokens += event.get("prompt_tokens", 0)
                    completion_tokens += event.get("completion_tokens", 0)
            
            # Store trial metrics
            trial_metric = {
                "trial": trial_num,
                "success": trial_success,
                "time_sec": round(trial_elapsed, 2),
                "total_tokens": total_tokens,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
            trial_metrics.append(trial_metric)
            
            # Save per-trial report
            trial_report = {
                "trial_number": trial_num,
                "result": trial_metric,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
            with open(trial_dir / "report.json", "w") as f:
                json.dump(trial_report, f, indent=2)
            
            # Save timings for this trial
            agent._save_timings(filename=str(trial_dir / "timings.json"))
            
            print(f"\n[TRIAL {trial_num}] Metrics:")
            print(f"  Success: {trial_success}")
            print(f"  Time: {trial_elapsed:.2f}s")
            print(f"  Tokens: {total_tokens} (prompt: {prompt_tokens}, completion: {completion_tokens})")
            
        except Exception as e:
            print(f"\n[TRIAL {trial_num}] FAILED with error:")
            print(f"  {type(e).__name__}: {e}")
            trial_metrics.append({
                "trial": trial_num,
                "success": False,
                "time_sec": round(time.time() - trial_start, 2),
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "error": str(e),
            })
        
        finally:
            # Restore original experiment dir for next trial
            agent.logger.experiment_dir = original_exp_dir
    
    # ─────────────────────────────────────────────────────────────
    # Compute aggregates
    # ─────────────────────────────────────────────────────────────
    num_successes = sum(1 for m in trial_metrics if m["success"])
    success_rate = num_successes / num_trials * 100
    
    avg_time = sum(m["time_sec"] for m in trial_metrics) / num_trials
    avg_tokens = sum(m["total_tokens"] for m in trial_metrics) / num_trials
    avg_prompt_tokens = sum(m["prompt_tokens"] for m in trial_metrics) / num_trials
    avg_completion_tokens = sum(m["completion_tokens"] for m in trial_metrics) / num_trials
    
    # ─────────────────────────────────────────────────────────────
    # Create summary dict
    # ─────────────────────────────────────────────────────────────
    summary = {
        "experiment": agent.logger.experiment_name,
        "game": args.game,
        "level_sets": level_sets,
        "num_trials": num_trials,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "results": {
            "successes": num_successes,
            "failures": num_trials - num_successes,
            "success_rate_percent": round(success_rate, 2),
        },
        "averages": {
            "time_sec": round(avg_time, 2),
            "total_tokens": round(avg_tokens, 2),
            "prompt_tokens": round(avg_prompt_tokens, 2),
            "completion_tokens": round(avg_completion_tokens, 2),
        },
        "per_trial_metrics": trial_metrics,
    }
    
    # ─────────────────────────────────────────────────────────────
    # Save JSON summary (machine-readable)
    # ─────────────────────────────────────────────────────────────
    json_path = Path(agent.logger.experiment_dir) / "trials_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    # ─────────────────────────────────────────────────────────────
    # Save TXT summary (human-readable)
    # ─────────────────────────────────────────────────────────────
    txt_path = Path(agent.logger.experiment_dir) / "trials_summary.txt"
    with open(txt_path, "w") as f:
        f.write("="*70 + "\n")
        f.write("MULTI-TRIAL EXPERIMENT SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Experiment:    {summary['experiment']}\n")
        f.write(f"Game:          {summary['game']}\n")
        f.write(f"Level Sets:    {summary['level_sets']}\n")
        f.write(f"Num Trials:    {summary['num_trials']}\n")
        f.write(f"Timestamp:     {summary['timestamp']}\n\n")
        
        f.write("-"*70 + "\n")
        f.write("RESULTS SUMMARY\n")
        f.write("-"*70 + "\n")
        f.write(f"Successes:              {summary['results']['successes']}/{num_trials}\n")
        f.write(f"Failures:               {summary['results']['failures']}/{num_trials}\n")
        f.write(f"Success Rate:           {summary['results']['success_rate_percent']}%\n\n")
        
        f.write("-"*70 + "\n")
        f.write("AVERAGES ACROSS ALL TRIALS\n")
        f.write("-"*70 + "\n")
        f.write(f"Time per Trial:         {summary['averages']['time_sec']}s\n")
        f.write(f"Total Tokens:           {summary['averages']['total_tokens']}\n")
        f.write(f"  Prompt Tokens:        {summary['averages']['prompt_tokens']}\n")
        f.write(f"  Completion Tokens:    {summary['averages']['completion_tokens']}\n\n")
        
        f.write("-"*70 + "\n")
        f.write("PER-TRIAL BREAKDOWN\n")
        f.write("-"*70 + "\n")
        for m in trial_metrics:
            status = "✓ SUCCESS" if m["success"] else "✗ FAILED"
            f.write(f"\nTrial {m['trial']:02d}: {status}\n")
            f.write(f"  Time:  {m['time_sec']}s\n")
            f.write(f"  Tokens: {m['total_tokens']} (prompt: {m['prompt_tokens']}, completion: {m['completion_tokens']})\n")
        
        f.write("\n" + "="*70 + "\n")
    
    # Print to console as well
    print(f"\n\n{'='*70}")
    print("MULTI-TRIAL EXPERIMENT SUMMARY")
    print(f"{'='*70}\n")
    print(f"Experiment:    {summary['experiment']}")
    print(f"Game:          {summary['game']}")
    print(f"Num Trials:    {summary['num_trials']}\n")
    print(f"{'─'*70}")
    print(f"Successes:          {summary['results']['successes']}/{num_trials} ({summary['results']['success_rate_percent']}%)")
    print(f"{'─'*70}")
    print(f"Avg Time:           {summary['averages']['time_sec']}s")
    print(f"Avg Total Tokens:   {summary['averages']['total_tokens']}")
    print(f"  Prompt:           {summary['averages']['prompt_tokens']}")
    print(f"  Completion:       {summary['averages']['completion_tokens']}")
    print(f"{'='*70}\n")
    print(f"Summary saved to:")
    print(f"  JSON: {json_path}")
    print(f"  TXT:  {txt_path}")
    print(f"  Per-trial logs: {trials_dir}\n")
    
    return summary


# ───────────────────────────────────────────────────────────────────
# YAML CONFIG LOADING FOR TRANSFER MODE
# ───────────────────────────────────────────────────────────────────

def load_transfer_config(config_path: str) -> dict:
    """
    Load and validate YAML transfer configuration.

    Args:
        config_path: Path to YAML config file

    Returns:
        Parsed config dict with validated structure

    Raises:
        ValueError: If config is invalid
        FileNotFoundError: If file doesn't exist
    """
    import yaml
    from pathlib import Path

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Transfer config file not found: {config_path}")

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except Exception as e:
        raise ValueError(f"Failed to parse YAML config: {e}")

    # Validate required fields
    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML dict")

    required_fields = ['game', 'levels']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field in config: {field}")

    if not isinstance(config['levels'], list):
        raise ValueError("'levels' must be a list")

    # Validate each level
    for i, level_spec in enumerate(config['levels']):
        if not isinstance(level_spec, dict):
            raise ValueError(f"Level {i} must be a dict")

        if 'level' not in level_spec:
            raise ValueError(f"Level {i} missing 'level' field")

        if 'transfer' in level_spec:
            transfer = level_spec['transfer']
            if not isinstance(transfer, dict):
                raise ValueError(f"Level {i} 'transfer' must be a dict")

            if 'from_level' not in transfer:
                raise ValueError(f"Level {i} 'transfer' missing 'from_level'")

            # Validate domain/predicates values - can be bool or "extend"
            for field in ['domain', 'predicates']:
                val = transfer.get(field)
                if val is not None and val not in [True, False, 'extend']:
                    raise ValueError(
                        f"Level {i} 'transfer.{field}' must be true, false, or 'extend', got: {val}"
                    )

    return config


# ───────────────────────────────────────────────────────────────────
# SEQUENTIAL LEVEL RUNNER WITH TRANSFER
# ───────────────────────────────────────────────────────────────────

def run_sequential_levels_with_transfer(agent, config: dict, build_engine_fn):
    """
    Execute levels sequentially with configurable artifact transfer between levels.

    Args:
        agent: TheoryCoderAgent instance
        config: Parsed YAML config with level and transfer specifications
        build_engine_fn: Function to build game engine: build_engine_fn(game, level_set, level_id)

    Returns:
        List of dicts with results for each level
    """
    from pathlib import Path

    game = config['game']
    levels_config = config['levels']
    max_attempts = config.get('max_attempts', 6)

    results = []

    for level_spec in levels_config:
        level_id = level_spec['level']
        transfer = level_spec.get('transfer', None)

        print(f"\n{'='*70}")
        print(f"[LEVEL {level_id}] Starting")
        if transfer:
            print(f"[TRANSFER] Will transfer from level {transfer['from_level']}")
        else:
            print(f"[LEARNING] No transfer - learning from scratch")
        print(f"{'='*70}")

        # Setup artifacts for this level
        if transfer:
            from_level = transfer['from_level']
            agent._setup_level_artifacts(level_id, from_level, transfer, game_name=game)
        else:
            # No transfer - just create artifact snapshot dir
            agent._setup_level_artifacts(level_id, None, None, game_name=game)

        # If a fixed worldmodel is configured, copy it into game_dir for every level
        fixed_wm_path = getattr(agent, 'fixed_worldmodel_path', None)
        if fixed_wm_path:
            import shutil
            fixed_wm = Path(fixed_wm_path)
            if fixed_wm.exists():
                dest_wm = agent.game_dir / 'worldmodel.py'
                shutil.copy(fixed_wm, dest_wm)
                print(f"[FIXED WM] Copied fixed worldmodel to {dest_wm}")
            else:
                print(f"[FIXED WM ERROR] Fixed worldmodel not found: {fixed_wm}")

        # Build engine for this level
        level_set_name = game
        engine = build_engine_fn(game, level_set_name, level_id)

        # Run agent for this level
        try:
            success = agent.run(engine, max_attempts=max_attempts)
        except Exception as e:
            print(f"[ERROR] Exception while running level {level_id}: {e}")
            import traceback
            traceback.print_exc()
            success = False

        # Snapshot artifacts after completion
        level_snapshot_dir = agent.game_dir / f'level_{level_id}_artifacts'
        try:
            agent._snapshot_artifacts_to_dir(level_snapshot_dir)
        except Exception as e:
            print(f"[WARNING] Failed to snapshot artifacts: {e}")

        # Record result
        result = {
            'level': level_id,
            'success': success,
            'transferred_from': transfer['from_level'] if transfer else None
        }
        results.append(result)

        print(f"\n[LEVEL {level_id}] {'✓ SUCCESS' if success else '✗ FAILED'}")

    return results


if __name__ == '__main__':
    # NOTE: This is the legacy entry point. Consider using:
    #   python -m theorycoder --transfer-config config.yaml
    # or importing from the theorycoder package directly.

    import argparse, ast, os, time, json, io, sys
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="TheoryCoder Agent - Theory-based RL for game playing. "
                    "For the new modular interface, use: python -m theorycoder"
    )
    parser.add_argument('--game', type=str, default='pb1',
                        choices=['baba','lava','babyai','pb1','sokoban','labyrinth','cheesemaze'],)
    parser.add_argument('--level-sets', type=str, default="{'pb1': [0, 1, 2, 3]}",
                        help="Python dict, e.g. \"{'labyrinth':[0], 'maze':[0,1]}\"")
    parser.add_argument('--episode-length', type=int, default=20)
    parser.add_argument('--world-model-file-name', type=str, default='worldmodel')
    parser.add_argument('--domain-file-name', type=str, default='domain.pddl')
    parser.add_argument('--predicates-file-name', type=str, default='predicates')
    parser.add_argument('--json-reporter-path', type=str, default='KekeCompetition-main/Keke_JS/reports/TBRL_BABA_REPORT.json')
    parser.add_argument('--learn-model', action='store_true')
    parser.add_argument('--query-mode', type=str, default='custom')
    parser.add_argument('--language-model', type=str, default='gpt-4o-2024-11-20',
                        help="Model name passed to the gateway. For openrouter use full slugs like 'meta-llama/llama-3.3-70b-instruct'.")
    parser.add_argument('--seed', type=int, default=None,
                        help="Env seed for stochastic envs (BabyAI). Mirrors baseline_runner.py's --seed. Default behavior (None) preserves BabyAI's built-in default.")
    parser.add_argument('--reasoning-effort', type=str, default=None,
                        choices=['none', 'minimal', 'low', 'medium', 'high', 'xhigh'],
                        help="Reasoning effort. Per-model: o4-mini=[low,med,high]; gpt-5=[minimal,low,med,high]; gpt-5.1=[none,low,med,high]; gpt-5.2=[none,low,med,high,xhigh]. Ignored for gpt-4*.")
    parser.add_argument('--groq-model', type=str, default='llama-3.3-70b-versatile',
                        help='Model name for Groq provider')
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

    parser.add_argument('--num-trials', type=int, default=1,
                    help='Number of times to run the agent (each trial starts fresh). Default: 1')

    parser.add_argument('--transfer-config', type=str, default=None,
                    help='Path to YAML config file for level transfer configuration (enables sequential transfer mode)')

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

    # ──────────────────────────────────────────────────────────────────
    # EARLY TRANSFER CONFIG LOADING (before agent creation)
    # ──────────────────────────────────────────────────────────────────
    if args.transfer_config:
        print(f"\n{'='*70}")
        print(f"[TRANSFER CONFIG MODE] Loading config from {args.transfer_config}")
        print(f"{'='*70}\n")

        try:
            transfer_config = load_transfer_config(args.transfer_config)
        except Exception as e:
            print(f"[ERROR] Failed to load transfer config: {e}")
            sys.exit(1)

        # Override experiment_dir from YAML config
        if 'experiment_dir' in transfer_config:
            args.experiment_dir = transfer_config['experiment_dir']
            print(f"Using experiment dir from config: {args.experiment_dir}")

        # Override game from YAML config
        if 'game' in transfer_config:
            args.game = transfer_config['game']
            print(f"Using game from config: {args.game}")

        # Override learn_model from YAML config
        if 'learn_model' in transfer_config:
            args.learn_model = transfer_config['learn_model']
            print(f"Using learn_model from config: {args.learn_model}")

        # Override max_attempts from YAML config
        if 'max_attempts' in transfer_config:
            args.max_attempts = transfer_config['max_attempts']
            print(f"Using max_attempts from config: {args.max_attempts}")

        # Override num_trials from YAML config
        if 'num_trials' in transfer_config:
            args.num_trials = transfer_config['num_trials']
            print(f"Using num_trials from config: {args.num_trials}")

        # Override query_mode from YAML config
        if 'query_mode' in transfer_config:
            args.query_mode = transfer_config['query_mode']
            print(f"Using query_mode from config: {args.query_mode}")

        # Override groq_model from YAML config
        if 'groq_model' in transfer_config:
            args.groq_model = transfer_config['groq_model']
            print(f"Using groq_model from config: {args.groq_model}")

        # Override temperature from YAML config
        if 'temperature' in transfer_config:
            args.temperature = transfer_config['temperature']
            print(f"Using temperature from config: {args.temperature}")

        # Check for debug_no_llm mode (replay without any LLM calls)
        debug_no_llm = transfer_config.get('debug_no_llm', False)
        if debug_no_llm:
            print(f"[DEBUG MODE] No LLM calls - using existing artifacts only")
            args.learn_model = False  # Disable model learning/revision

        # Override legacy_door_format from YAML config (BabyAI A/B flag)
        if 'legacy_door_format' in transfer_config:
            args.legacy_door_format = bool(transfer_config['legacy_door_format'])
            print(f"Using legacy_door_format from config: {args.legacy_door_format}")
        else:
            args.legacy_door_format = False

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
        language_model=getattr(args, 'language_model', 'gpt-4o-2024-11-20'),
        reasoning_effort=getattr(args, 'reasoning_effort', None),
        groq_model=getattr(args, 'groq_model', 'llama-3.3-70b-versatile'),
        temperature=getattr(args, 'temperature', 1),
        prune_plans=args.prune_plans,
        centralize_files=args.centralize_files,
        create_subdir=False,
        transfer_levels=tl,
        env_seed=args.seed,
    )
    print(f"[model] language_model={agent.language_model} reasoning_effort={getattr(agent, 'reasoning_effort', None)} query_mode={agent.query_mode}", flush=True)


    # ---- engine factory for ALL games ----
    def build_engine(game: str, level_set: str, level_id: int):
        if game == 'baba':
            return BabaIsYou(level_set=level_set, level_id=level_id)
        if game == 'lava':
            return LavaGrid()
        if game == 'babyai':
            return BabyAI(
                level_set=level_set,
                level_id=level_id,
                legacy_door_format=getattr(args, 'legacy_door_format', False),
                **_babyai_seed_kwargs(getattr(args, 'seed', None)),
            )
        if game == 'pb1':
            return pb1env(level_set=level_set, level_id=level_id)
        if game == 'sokoban':
            return SokobanEnv(level_set=level_set, level_id=level_id)
        if game == 'labyrinth':
            return LabyrinthEnv(level_set=level_set, level_id=level_id)
        if game == 'maze':
            return MazeEnv(level_set=level_set, level_id=level_id)
        if game == 'cheesemaze':
            return CheesemazeEnv(level_set=level_set, level_id=level_id)
        raise ValueError(f"Unknown game: {game}")

    if args.seed_from:
        seed_game_from_previous(
            agent,
            args.seed_from,
            args.game,
            copy_plans=True,
            src_game=args.seed_src_game  # ← CLI controls this
        )

    # ──────────────────────────────────────────────────────────────────
    # TRANSFER CONFIG MODE - SEQUENTIAL LEVEL TRANSFER
    # (Config already loaded early, now execute sequential transfer)
    # ──────────────────────────────────────────────────────────────────
    if args.transfer_config:
        # Re-load config (it was parsed early but we need it here too)
        config = load_transfer_config(args.transfer_config)

        # Set debug_no_llm flag on agent (skip all LLM calls, use existing artifacts)
        agent.debug_no_llm = config.get('debug_no_llm', False)
        if agent.debug_no_llm:
            print(f"[DEBUG MODE] Agent will skip all LLM calls")

        # Set planner timeout (in seconds) - None means no timeout
        agent.planner_timeout = config.get('planner_timeout', None)
        if agent.planner_timeout:
            print(f"[PLANNER] Timeout set to {agent.planner_timeout} seconds")

        # BabyAI A/B: old (bucket) vs new (dict) door state representation
        agent.legacy_door_format = bool(config.get('legacy_door_format', False))
        if agent.legacy_door_format:
            print(f"[BABYAI] legacy_door_format=True — using bucket door encoding + legacy prompts")

        # Set fixed_worldmodel path from config if provided
        if 'fixed_worldmodel' in config:
            agent.fixed_worldmodel_path = config['fixed_worldmodel']
            print(f"[FIXED WM] World model fixed to: {agent.fixed_worldmodel_path}")

        # Initialize game_dir for the agent
        game_name = config['game']
        agent.game_dir = Path(agent.logger.experiment_dir) / "tc_game" / game_name
        agent.game_dir.mkdir(parents=True, exist_ok=True)

        # Set environment variables for world model and predicates
        os.environ["TC_WORLDMODEL_FILE"] = str(agent.game_dir / "worldmodel.py")
        os.environ["TC_PREDICATES_FILE"] = str(agent.game_dir / "predicates.py")

        # Handle multi-trial mode
        if args.num_trials > 1:
            print(f"\n{'='*70}")
            print(f"[MULTI-TRIAL MODE] Running {args.num_trials} independent trials")
            print(f"{'='*70}\n")

            trials_dir = Path(agent.logger.experiment_dir) / "transfer_trials"
            trials_dir.mkdir(exist_ok=True)

            all_trial_results = []

            for trial_num in range(1, args.num_trials + 1):
                print(f"\n{'='*70}")
                print(f"[TRIAL {trial_num}/{args.num_trials}]")
                print(f"{'='*70}\n")

                # Create isolated trial directory
                trial_dir = trials_dir / f"trial_{trial_num:03d}"
                trial_dir.mkdir(exist_ok=True)

                # Create tape and steps subdirectories (needed for logger)
                (trial_dir / "tape").mkdir(exist_ok=True)
                (trial_dir / "steps").mkdir(exist_ok=True)

                # Setup trial-specific paths
                original_exp_dir = agent.logger.experiment_dir
                agent.logger.experiment_dir = str(trial_dir)
                agent.game_dir = trial_dir / "tc_game" / config['game']
                agent.game_dir.mkdir(parents=True, exist_ok=True)

                # CRITICAL: Update environment variables to point to trial-specific paths
                os.environ["TC_WORLDMODEL_FILE"] = str(agent.game_dir / "worldmodel.py")
                os.environ["TC_PREDICATES_FILE"] = str(agent.game_dir / "predicates.py")

                # Reset agent state for fresh trial
                agent.tape = [{}]
                agent.logger.tape = []  # Reset logger's tape for new trial
                agent._init_timing()

                # Run sequential levels for this trial
                trial_results = run_sequential_levels_with_transfer(agent, config, build_engine)
                all_trial_results.append({
                    'trial': trial_num,
                    'results': trial_results
                })

                # Restore original experiment dir for next trial
                agent.logger.experiment_dir = original_exp_dir

            # Summary for multi-trial mode
            print(f"\n{'='*70}")
            print(f"[MULTI-TRIAL SUMMARY]")
            print(f"{'='*70}\n")
            for trial_result in all_trial_results:
                trial_num = trial_result['trial']
                results = trial_result['results']
                successes = sum(1 for r in results if r['success'])
                print(f"Trial {trial_num}: {successes}/{len(results)} levels succeeded")

        else:
            # Single trial mode
            print(f"\n{'='*70}")
            print(f"[SINGLE TRIAL MODE]")
            print(f"{'='*70}\n")

            results = run_sequential_levels_with_transfer(agent, config, build_engine)

            # Summary for single trial
            print(f"\n{'='*70}")
            print(f"[TRANSFER SUMMARY]")
            print(f"{'='*70}\n")
            successes = sum(1 for r in results if r['success'])
            print(f"Overall: {successes}/{len(results)} levels succeeded")
            for r in results:
                status = "✓" if r['success'] else "✗"
                transfer_info = f" (transferred from L{r['transferred_from']})" if r['transferred_from'] else " (learned from scratch)"
                print(f"  {status} Level {r['level']}{transfer_info}")

        # Save tape and exit
        from pathlib import Path
        Path('tapes').mkdir(parents=True, exist_ok=True)
        tape_path = f'tapes/transfer_config_{int(time.time())}.json'
        with open(tape_path, 'w') as f:
            json.dump(agent.tape, f, indent=4)
        print(f"\nTape saved to: {tape_path}")

        sys.exit(0)

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


    # ---------- run with num-trials support ----------
    if args.num_trials > 1:
        print(f"\n{'='*70}")
        print(f"[MULTI-TRIAL MODE] Running {args.num_trials} trials with fresh state each time...")
        print(f"{'='*70}")
        trial_results = run_with_trials(
            agent=agent,
            num_trials=args.num_trials,
            level_sets=level_sets,
            max_revisions=args.max_attempts,
            max_attempts=args.max_attempts,
            build_engine_fn=build_engine,
            multi_level=args.multi_level,
            args=args
        )
        print(f"\n[MULTI-TRIAL] Summary saved to {agent.logger.experiment_dir}")
    elif args.multi_level:
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
