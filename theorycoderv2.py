"""
"""
import importlib
from pathlib import Path
from copy import deepcopy
import json
import os
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
# from games import LavaGrid, BabaIsYou
import ast
from levelrunner import actor
import inspect
import utils
from preprocessing import *
import openai
from groq import Groq
from experiment_logger import ExperimentLogger
from time import strftime, gmtime
from babyai_env import BabyAI
import minigrid
# from boulderdash2_env import Boulderdash2Env
# from pb1_env import pb1env
# from sokoban_env import SokobanEnv
from games import BabaIsYou
# from clusterbox_env import ClusterboxEnv

from typing import List
import shutil

import subprocess
from subprocess import CalledProcessError

initialize_world_model_prompt = \
"""You are an AI agent that must come up with a transition model of the game you are playing. 

A BFS low-level planner that will use your synthesized transition model to find the low-level actions that will allow you to win levels of the game.

You are also given state transition after executing random actions that will help as well.
Note that if there is no change returned after doing that action, it means that moving was prevented somehow such as by an obstacle. 

The levels you start out with will be simpler but you will be adding on more and more as time progresses. 
So try to make the transition model general and avoid hardcoding anything from the state dictionary keys. Feel free to infer the types of interactions that will occur in later levels. 
Do not feel like you need to build the transition model for just this replay buffer. 

DESCRIPTION OF DOMAIN:

This domain is a game where you move around a red agent but cannot move onto walls. 

You need to face in the direction of an item
and stand one square adjacent from it in order to pick it up and have it in your inventory. You can only hold one item at a time.
You can pickup boxes, balls, keys. 

Doors can be opened in this game by standing adjacent one square from them and facing towards them.
Once a door is opened you can stand on the same position as it and move past it. 

You can open locked doors by having the corresponding color key for that door. Once you use the key to unlock the door,
the key will remain in your inventory.

Once a door is opened it will remain on the map and be an opened_ door.

For example, consider a map where there is a closed_red_door (this means its not locked). 
When you toggle it, it will turn into an open_red_door

You may also encounter a locked_red_door and when you toggle it with a key in your inventory it'll become an open_red_door

If you choose to drop an item from your inventory you will drop it in the position right in front of you in the direction you are facing.
You cannot drop an item in front of you if another item is in that square.

Furthermore, you SHOULD NOT DROP the ITEM any 1 square away from a door way or else it will block it.

Another thing to note is that boxes can be picked up AND they can also be toggled. If you toggle a box, the corresponding
color key will appear in that square and the box will be removed from the state.

For example if you toggle a box, then a key will appear at the same coordinate that the box is at.

However, this ONLY happens when there is a locked door of that color and a key of the same color as the locked door does NOT exist in the state.

Effectively, you can think of the box turning into the corresponding key in a way. 

So for example purple_box at [2, 5] will turn into purple_key at [2, 5] replacing that in that state 

or in other words purple_box will be deleted from the map completely.

purple_box -> purple_key



NOTES: 

Also, remember to return the state if you modified it directly or return the new_state if you deepcopied it.

Remember to make your world model general you can use string checks like .endswith or .beginswith instead of hardcoding a list of entities in your world model.

You can replace the keys of any state dictionary items that transform!

YOU MUST ATTEMPT TO MODEL ALL THE MECHANICS FROM THE ACTION SPACE. 

CURRENT STATE:

{current_state}

ACTION SPACE:

{actions_set}

Replay Buffer (last {num_random_actions} transitions):

{errors_from_world_model}

UTILS:

{utils}


RESPONSE FORMAT:

- Make sure that you return the correct state for example if you made a deepcopy of the state and modify the deep copy then return the new_state
- If you modify the state directly then return the state instead of new_state
- Try to generalize your world model. For example, do not just assume that all entities will be the same color in all variations
- Your world model will be used in other variations where entities will be of different colors, but the principles should remain the same
- For example, you can use string checks like .endswith instead of hardcoding a list of entities in your world model.

```python

# make sure to include these import statements
from copy import deepcopy
from utils import directions

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

This domain is a game where you move around a red agent but cannot move onto walls. 

You need to face in the direction of an item
and stand one square adjacent from it in order to pick it up and have it in your inventory. You can only hold one item at a time.
You can pickup boxes, balls, keys. 

Doors can be opened in this game by standing adjacent one square from them and facing towards them.
Once a door is opened you can stand on the same position as it and move past it. 

You can open locked doors by having the corresponding color key for that door. Once you use the key to unlock the door,
the key will remain in your inventory.

Once a door is opened it will remain on the map and be an opened door.

For example, consider a map where there is a closed_red_door (this means its not locked). 
When you toggle it, it will turn into an open_red_door

You may also encounter a locked_red_door and when you toggle it with a key in your inventory it'll become an open_red_door

If you choose to drop an item from your inventory you will drop it in the position right in front of you in the direction you are facing.
You cannot drop an item in front of you if another item is in that square.

Furthermore, you SHOULD NOT DROP the ITEM any 1 square away from a door way or else it will block it.

Another thing to note is that boxes can be picked up AND they can also be toggled. If you toggle a box, the corresponding
color key will appear in that square and the box will be removed from the state.

For example if you toggle a box, then a key will appear at the same coordinate that the box is at. 

However, this ONLY happens when there is a locked door of that color and a key of the same color as the locked door does NOT exist in the state.

Effectively, you can think of the box turning into the corresponding key in a way.

So for example purple_box at [2, 5] will turn into purple_key at [2, 5] replacing that in that state

or in other words purple_box will be deleted from the map completely.

purple_box -> purple_key


NOTES:

Feel free to also explain your thinking outside of the markup tags, but know that I will only use the code inside the markup tags. 

The exploration plans are set up to help guide you to your overall goal. 

You can replace the keys of any state dictionary items that transform!


YOUR OVERALL GOAL TO WIN THIS LEVEL:

{mission_goal}

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
- Try to generalize your world model. For example, do not just assume that all entities will be the same color in all variations
- Your world model will be used in other variations where entities will be of different colors, but the principles should remain the same
- For example, you can use string checks like .endswith or .beginswith instead of hardcoding a list of entities in your world model.

```Python

# make sure to include these import statements
from predicates import *
from copy import deepcopy
from utils import directions

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
from predicates import *
from copy import deepcopy
from utils import directions

def transition_model(state, action):


	Return State

```

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



import sys
# ensure your project root is first on sys.path
proj_root = os.path.abspath(os.path.dirname(__file__))
if proj_root in sys.path:
    sys.path.remove(proj_root)
sys.path.insert(0, proj_root)


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
        query_mode='groq',  # Options: 'langchain_openai', 'openai_direct', 'groq'
        groq_model="llama3-8b-8192",  # Specify the Groq model
        # language_model='gpt-4-turbo-preview',
        temperature=0.7,
        # temperature=1,
        episode_length=20,
        do_revise_model=False,
        sparse_interactions=True,  # Only run subset of world model
        observation_memory_size=1,
        planner_explore_prob=0,
        max_replans=1,
        plans_file_name='plans.json',  # Default to a generic file if not specified
        base_dir=None,  # Added for experiment logging
        experiment_name=None,  # Added for experiment logging
        prune_plans=False,  # Add this parameter to switch between methods
    ):

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
        else:
            raise ValueError(f"Unsupported query_mode: {query_mode}")

        # I/O
        self.world_model_save_name = '_model_tmp'
        self.world_model_load_name = world_model_load_name  # Possibly load existing model
        self.operators_save_name = '_operators_tmp'
        self.operators_load_name = operators_load_name
        self.predicates_save_name = '_predicates_tmp'
        self.predicates_load_name = predicates_load_name
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


         # Load domain PDDL and predicates files
        self._load_predicates(self.predicates_file_name)
        
        self.load_utils()

        # Add new runtime variables to track exploratory plans
        self.runtime_vars['exploratory_plans'] = []
        self.runtime_vars['unsatisfied_preconditions'] = []

        # Initialize experiment logger
        self.logger = ExperimentLogger(base_dir or os.getcwd(), experiment_name)

        # Initialize level statistics
        self.level_statistics = {}

        self.prune_plans = prune_plans
        self.aggregated_dataset = []  # Initialize the aggregated dataset

    def query_lm(self, prompt):
        """
        Query the LLM based on the selected query mode.
        Supports LangChain OpenAI, OpenAI direct API, and Groq.
        """
        # if not isinstance(prompt, str):
        #     raise ValueError("Prompt must be a string.")

        if self.query_mode == 'langchain_openai':
            # Query using LangChain OpenAI
            chat_prompt = HumanMessagePromptTemplate.from_template(prompt)
            return self.llm_client.invoke(chat_prompt.to_messages()).content

        elif self.query_mode == "openai_direct":
            # Ensure the prompt is correctly formatted for the OpenAI API
            messages = [{"role": "user", "content": prompt}]

            # Use the OpenAI client to send the request
            completion = self.llm_client.chat.completions.create(
                model=self.language_model,
                messages=messages,
                temperature=self.temperature,
                seed=42,  # Add the seed for reproducibility
            )
            # Return the generated response
            return completion.choices[0].message.content.strip(), completion if completion else ""

        elif self.query_mode == 'groq':
            # Query using Groq API
            messages = [{"role": "user", "content": prompt}]  # Format for Groq
            response = self.llm_client.chat.completions.create(
                messages=messages,
                model=self.groq_model
            )
            return response.choices[0].message.content.strip()

        else:
            raise ValueError(f"Unsupported query_mode: {self.query_mode}")



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


    def capture_world_model(self):
        # Path to the worldmodel.py file
        world_model_path = "worldmodel.py"
        
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
        text = Path(f"abstraction_prompts/{name}.txt").read_text()
        return text.format(**kwargs)
    

    def extract_code_block(self, text: str, lang: str, which: int) -> str:
        pattern = rf"```{lang}(.*?)(?=```)"
        blocks = re.findall(pattern, text, re.DOTALL)
        return blocks[which-1].strip() if len(blocks) >= which else ""

    def _solve_existing_pddl(self, level: int):
        """
        Attempt to solve an existing domain/problem pair with Fast-Downward.
        If it fails with exit code 12 (no solution), regenerate domain/problem via LLM.
        """
        game_name = self._get_game_name()
        domain_path = self.game_dir / f"{game_name}_domain.pddl"
        problem_path = self.game_dir / f"{game_name}_{level}.pddl"

        if domain_path.exists() and problem_path.exists():
            print(f"[{game_name}] domain & problem exist → running Fast-Downward")

            cmd = [
                "python", "/home/z/downward/fast-downward.py",
                str(domain_path), str(problem_path),
                "--search", "astar(blind())"
            ]

            try:
                subprocess.run(cmd, check=True)
            except CalledProcessError as e:
                print(f"Fast Downward failed with exit code {e.returncode}")
                print("Falling back to regenerating PDDL...")
                return self.generate_and_solve_pddl(level=level)


            plan = self.parse_sas_plan("sas_plan")
            # Format plan as hierarchical subplans for level "0"
            # plan_dict = { str(level): [f"move_to {step}" for step in plan] }

            plan_dict = { str(level): plan }
            plan_path = self.game_dir / f"{game_name}_plans.json"
            with open(plan_path, "w") as f:
                json.dump(plan_dict, f, indent=2)



            print(f"[{game_name}] plan written to {plan_path}")
            return plan

        return None



    def generate_and_solve_pddl(self, level: int):
        """
        1) Write domain & problem into self.game_dir
        2) Run Fast-Downward; on failure, ask the LLM to regen PDDL via
           abstraction_prompts/regen_pddl_files.txt and retry once
        """
        folder = self.game_dir
        folder.mkdir(parents=True, exist_ok=True)

        raw = json.dumps(self.engine.get_obs())

        # 1) INITIAL GENERATION via existing init prompt
        prompt = self.load_prompt("init_pddl_files", raw_state=raw)
        response, token_info = self.query_lm(prompt)

        step_dir = Path(self.logger.create_step("pddl_init"))


        # create a new “pddl_init” step
        # dump out both the prompt, the response, and the raw metadata
        with open(os.path.join(step_dir, "prompt.txt"), "w") as f:
            f.write(prompt)
        with open(os.path.join(step_dir, "response.txt"), "w") as f:
            f.write(response)
        with open(os.path.join(step_dir, "completion_info_pddl_init.json"), "w") as f:
            # if token_info is a langchain or groq object you might .dict() it first
            json.dump(token_info, f, default=lambda o: getattr(o, "to_dict", lambda: str(o))(), indent=2)


        # extract and write out
        domain_code  = self.extract_code_block(response, "pddl", which=1)
        problem_code = self.extract_code_block(response, "pddl", which=2)

        game_name    = self._get_game_name()
        domain_path  = folder / f"{game_name}_domain.pddl"
        problem_path = folder / f"{game_name}_{level}.pddl"

        domain_path.write_text(domain_code)
        problem_path.write_text(problem_code)

        # copy the exact files into the step directory
        shutil.copy(str(domain_path), str(step_dir / domain_path.name))
        shutil.copy(str(problem_path), str(step_dir / problem_path.name))

        # 2) CALL FAST-DOWNWARD

        cmd = [
            "python", "/home/z/downward/fast-downward.py",
            str(domain_path),
            str(problem_path),
            "--search", "astar(blind())"
        ]
        try:
            subprocess.run(cmd, check=True)
        except CalledProcessError:
            # FAILED → ask LLM to regen simpler PDDL
            regen_tpl = Path("abstraction_prompts/regen_pddl_files.txt").read_text()
            regen_prompt = regen_tpl.format(
                domain_file=domain_code,
                problem_file=problem_code,
                raw_state=raw
            )
            # after
            regen_text, regen_meta = self.query_lm(regen_prompt)
            new_dom  = self.extract_code_block(regen_text, "pddl", 1)
            new_prb  = self.extract_code_block(regen_text, "pddl", 2)

            step_dir = Path(self.logger.create_step("pddl_regen"))


            # after regen_text, regen_meta = self.query_lm(regen_prompt)
            with open(os.path.join(step_dir, "completion_info_pddl_regen.json"), "w") as f:
                json.dump(regen_meta, f, default=lambda o: getattr(o, "to_dict", lambda: str(o))(), indent=2)


            domain_path.write_text(new_dom)
            problem_path.write_text(new_prb)

            # copy regenerated files
            shutil.copy(str(domain_path), str(step_dir / domain_path.name))
            shutil.copy(str(problem_path), str(step_dir / problem_path.name))


            # retry exactly once
            subprocess.run(cmd, check=True)

        # 3) PARSE PLAN
        plan = self.parse_sas_plan("sas_plan")
        plan_dict = { str(level): plan }  # store only the plan under the level number
        plan_path = self.game_dir / f"{game_name}_plans.json"
        with open(plan_path, "w") as f:
            json.dump(plan_dict, f, indent=2)


        return plan

    def generate_and_save_predicates(self, domain_path: Path, problem_path: Path):
        """
        If predicates.py is empty, call the LLM with
        abstraction_prompts/init_python_predicates.txt to synthesize it,
        then write it both to the shared predicates.py and into the
        per‐game folder.
        """
        tpl = Path("abstraction_prompts/init_python_predicates.txt").read_text()
        raw = json.dumps(self.engine.get_obs())
        prompt = tpl.format(
            domain_file=domain_path.read_text(),
            problem_file=problem_path.read_text(),
            raw_state=raw
        )

        # create a new “predicates_init” step
        step_dir = Path(self.logger.create_step("predicates_init"))

        resp, completion = self.query_lm(prompt)

        # save prompt, response, and completion metadata
        with open(step_dir / "prompt.txt", "w") as f:
            f.write(prompt)
        with open(step_dir / "response.txt", "w") as f:
            f.write(resp)
        with open(step_dir / "completion_info.json", "w") as f:
            json.dump(
                completion,
                f,
                default=lambda o: getattr(o, "to_dict", lambda: str(o))(),
                indent=2
            )

        # extract the first python code block
        predicates_code = self.extract_code_block(resp, "python", 1)

        # write into your shared module
        with open(f"{self.predicates_file_name}.py", "w") as f:
            f.write(predicates_code)
        # snapshot into the game folder
        (self.game_dir / "predicates.py").write_text(predicates_code)

        # copy the exact file into the step directory for audit
        shutil.copy(self.game_dir / "predicates.py", step_dir / "predicates.py")

        # update flags & runtime vars
        self.runtime_vars['predicates'] = predicates_code
        self.predicates_empty = False
        print("✅ Predicates synthesized and saved.")



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
                resp, completion = self.query_lm(prompt)
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
                        json.dump(
                                completion,
                                f,
                                default=lambda o: getattr(o, "to_dict", lambda: str(o))(),
                                indent=2
                            )

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
        obs = self.replay_buffers[::1]

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

    def _revise_world_model(self):
        if not self.do_revise_model:
            return

        self.tape[-1]['revision_prompts'] = {}
        self.tape[-1]['revision_responses'] = {}

        examples, error_count = self._choose_synthesis_examples()

        if isinstance(self.engine, BabyAI):
            mission = self.engine.mission

        if self._do_revise_model(error_count):
            prompt = self.revise_world_model_prompt.format(
                actions_set=self.engine.actions_set,
                errors_from_world_model='\n\n'.join(examples),
                world_model_str=self.runtime_vars['world_model_str'],
                utils="directions = {\n    'left': [-1, 0],\n    'right': [1, 0],\n    'up': [0, 1],\n    'down': [0, -1],\n}",
            )

            print(prompt)

            # Create step directory and save files
            step_dir = self.logger.create_step("revision")
            resp, completion = self.query_lm(prompt)
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
                    json.dump(
                                completion,
                                f,
                                default=lambda o: getattr(o, "to_dict", lambda: str(o))(),
                                indent=2
                            )

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

        examples, error_count = self._choose_synthesis_examples()

        prompt = self.initialize_world_model_prompt.format(
        current_state=self.runtime_vars['observations'][-1],
        actions_set=self.engine.actions_set,
        num_random_actions=num_actions,
        errors_from_world_model='\n\n'.join(examples),
        utils="directions = {\n    'left': [-1, 0],\n    'right': [1, 0],\n    'up': [0, 1],\n    'down': [0, -1],\n}"
    )

        file_name='current_prompt_INIT.txt'
        # Get the content of the first message in the prompt
        prompt_content = prompt

        print(prompt)

        # original_temp = self.temperature
        # original_llm = self.language_model
        # self.temperature = 1
        # self.language_model = "o1-preview"

        # Create or open the file and write the prompt content to it
        with open(file_name, 'w') as file:
            file.write(prompt_content)        
        resp, completion = self.query_lm(prompt)

        # self.temperature = original_temp
        # self.language_model = original_llm

        new_world_model_code = self.extract_code_from_response(resp)

        if new_world_model_code:
            # Update the world model string in runtime_vars
            self.runtime_vars['world_model_str'] = new_world_model_code

            # Overwrite the current worldmodel.py file with the new model
            self.overwrite_world_model(new_world_model_code)

            # Save the new model as an iteration (e.g., _iteration1)
            

            # Create step directory and save files
            step_dir = self.logger.create_step("initialize")
            self.logger.save_step_files(
                step_dir,
                prompt,
                resp,
                new_world_model_code,
                "worldmodel.py"
            )

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


            actionlist, state = actor(self.domain_file, subplan_exploratory, state, max_iterations=None, debug_callback=self._call_model_debug, level=None) #max its 2k original
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

                
                for subplan in self.plans.get(str(self.current_level), []):
                    action_seq, state = actor(self.domain_file, subplan, state, max_iterations=None, debug_callback=self._call_model_debug, level=self.current_level) #max its 2k original
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
        model_path = Path(f"{world_model_load_name}.py")
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
            print(f"World model file '{world_model_load_name}.py' not found.")
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
        predicates_path = Path(f"{predicates_file_name}.py")
        if predicates_path.exists():
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
        # shared (master) predicates.py
        with open(f"{self.predicates_save_name}.py", "w") as f:
            f.write(self.runtime_vars['predicates'])

        # snapshot in the game directory
        (self.game_dir / "predicates.py").write_text(self.runtime_vars['predicates'])



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
        # 1) Overwrite the master file so imports still work
        with open("worldmodel.py", "w") as f:
            f.write(new_code)

        # 2) Also save a copy inside the current game folder
        (self.game_dir / "worldmodel.py").write_text(new_code)


    def extract_code_from_response(self, response):
        # Use a regular expression to extract the Python code within ```python ``` tags (case-insensitive)
        code_match = re.search(r'```python(.*?)```', response, re.DOTALL | re.IGNORECASE)
        if code_match:
            return code_match.group(1).strip()
        else:
            return None

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
                # env.save_screen()
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
        from games import BabaIsYou, LavaGrid
        from babyai_env import BabyAI
        # from boulderdash2_env import Boulderdash2Env
        # from pb1_env import pb1env
        # from sokoban_env import SokobanEnv
        # from clusterbox_env import ClusterboxEnv


        if isinstance(self.engine, BabaIsYou):
            return "baba"
        elif isinstance(self.engine, LavaGrid):
            return "lava"
        elif isinstance(self.engine, BabyAI):
            return "babyai"
        elif isinstance(self.engine, Boulderdash2Env):
            return "boulderdash2"
        elif isinstance(self.engine, pb1env):
            return "pb1"
        elif isinstance(self.engine, SokobanEnv):
            return "sokoban"
        elif isinstance(self.engine, ClusterboxEnv):
            return "clusterbox"
        else:
            # fallback to the class name
            return self.engine.__class__.__name__.lower()


    def run(self, engine, max_revisions=5, max_attempts=6):
        self.engine = engine
        self.current_level = self.engine.level_id  # Or any other method to determine the level



        # --- insert per‐game directory setup here ---
        from pathlib import Path
        game_name   = self._get_game_name()
        self.game_dir = Path(f"tc_game/{game_name}")
        self.game_dir.mkdir(parents=True, exist_ok=True)

        # self.plans = self._load_plans()

         # — point domain_file at the per‐game file and load it —
        domain_path = "domain.pddl"
        self.domain_file = str(domain_path)
        self._load_domain_pddl(self.domain_file)

        # if self.domain_empty:
        #     print(f"[{game_name}] domain PDDL not found or empty → generating now")
        #     self.generate_and_solve_pddl(level=self.current_level)
        #     # ← insert predicate logic here
        #     self._load_domain_pddl(self.domain_file)

        #     self.plans = self._load_plans()
        #     print(f"Loaded plans for '{game_name}':", list(self.plans.keys()))
        # else:
        #     print(f"[{game_name}] domain PDDL already exists, skipping generation")
            # # try to solve it right now, using existing PDDL
            # plan = self._solve_existing_pddl(self.current_level)
            # if plan is None:
            #     print(f"[{game_name}] no problem PDDL found → you may need to generate it first")
        self.plans = self._load_plans()

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




                # — now re-check the shared predicates.py snapshot —
                self._load_predicates(self.predicates_file_name)
                if self.predicates_empty:
                    domain_path  = self.game_dir / f"{game_name}_domain.pddl"
                    problem_path = self.game_dir / f"{game_name}_{self.current_level}.pddl"
                    print("⚠️  predicates.py is empty → generating via LLM")
                    self.generate_and_save_predicates(domain_path, problem_path)

                # If the world model is not empty, proceed with the hierarchical planner
                mode = self._sample_planner_mode()  # Determine planner mode (explore/exploit)
                plan = self._hierarchical_planner(mode) 
                print("subplans from init model:", plan)
                self.reset(keep_model=True)
                model_was_revised = True

            else:
                # If the world model is not empty, proceed with the hierarchical planner
                mode = self._sample_planner_mode()  # Determine planner mode (explore/exploit)
                plan = self._hierarchical_planner(mode) 
                
    
            for action in plan:
                self.step_env(action)
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

                        # self.engine.save_screen(f"{self.logger.experiment_dir}/{self.current_level}_{attempt_count}.png")
                        # self.engine.save_screen()


                    return True

                # Check if the agent lost (e.g., died or failed critically)
                if self.engine.lost:
                    self.tape[-1]['exit_condition'] = 'lost'
                    self._update_solution(self.current_level, first_letters)
                    self.level_statistics[f"{self.engine.level_set}_{self.current_level}"]["first_letters"] = first_letters
                    self.level_statistics[f"{self.engine.level_set}_{self.current_level}"]["revisions"] = revision_count
                    self.level_statistics[f"{self.engine.level_set}_{self.current_level}"]["debugs"] = debug_count
                    print("AGENT DIED")
                    self._revise_world_model()
                    attempt_count += 1
                    model_was_revised = True
                    if isinstance(self.engine, BabyAI):
                        self.engine.close()

                        # self.engine.save_screen(self.logger.experiment_dir + 'level_lost')
                        # self.engine.save_screen(f"{self.logger.experiment_dir}/{self.current_level}_{attempt_count}.png")
                        # self.engine.save_screen()


                    break


            # If the model was revised, execute it first before proceeding with exploratory goals
            if model_was_revised:
                self.reset(keep_model=True)
                first_letters = ''  # Reset first_letters after model revision
                mode = self._sample_planner_mode()  # Determine planner mode (explore/exploit)
                plan = self._hierarchical_planner(mode) 
                for actions in plan:
                    self.step_env(action)
                    first_letters += action[0]  # Collect the first letters of each action


                    # Exit if agent won
                    if self.engine.won or (self.current_level == 6 and first_letters == 'rrruuu'):
                        self.tape[-1]['exit_condition'] = 'won'
                        self._update_solution(self.current_level, first_letters)
                        self.level_statistics[f"{self.engine.level_set}_{self.current_level}"]["first_letters"] = first_letters
                        self.level_statistics[f"{self.engine.level_set}_{self.current_level}"]["revisions"] = revision_count
                        self.level_statistics[f"{self.engine.level_set}_{self.current_level}"]["debugs"] = debug_count
                        print(first_letters)

                            # self.engine.save_screen(self.logger.experiment_dir + 'level_won')
                        # self.engine.save_screen(f"{self.logger.experiment_dir}/{self.current_level}_{attempt_count}.png")
                        # self.engine.save_screen()


                        return True

                    # Check if the agent lost (e.g., died or failed critically)
                    if self.engine.lost:
                        self.tape[-1]['exit_condition'] = 'lost'
                        self._update_solution(self.current_level, first_letters)
                        self.level_statistics[f"{self.engine.level_set}_{self.current_level}"]["first_letters"] = first_letters
                        self.level_statistics[f"{self.engine.level_set}_{self.current_level}"]["revisions"] = revision_count
                        self.level_statistics[f"{self.engine.level_set}_{self.current_level}"]["debugs"] = debug_count
                        print("AGENT DIED")
                        print(self.engine.get_obs())
                        attempt_count += 1

                            # self.engine.save_screen(self.logger.experiment_dir + 'level_lost')
                        # self.engine.save_screen(f"{self.logger.experiment_dir}/{self.current_level}_{attempt_count}.png")
                        # self.engine.save_screen()


                        break

            # Handle model revision if necessary
            if not self.is_world_model_empty() and self.do_revise_model and not model_was_revised:
                # exploratory_plans = self.propose_exploratory_plans(initial_state, self.domain_file)
                # print(exploratory_plans)
                # pruned_plans = self.prune_exploratory_plans(exploratory_plans)
                pruned_plans = ["collect_diamond diamond1","collect_diamond diamond2","collect_diamond diamond3","collect_diamond diamond4","collect_diamond diamond5","collect_diamond diamond6","collect_diamond diamond7","collect_diamond diamond8","collect_diamond diamond9","escape_via_exit avatar exitdoor"]
                print("pruned automatically", pruned_plans)


                LLM_pruned_plans = pruned_plans

                # # FOR DEBUGGING
                # # pruned_plans = ['open closed_red_door']
                # # if self.prune_plans
                # if len(pruned_plans) <= 3:
                #     print("Executing all pruned plans")
                #     LLM_pruned_plans = pruned_plans
                # elif len(pruned_plans) > 3 and self.prune_plans:
                #     # Prune the exploratory plans using LLM
                #     LLM_pruned_plans = self.prune_exploratory_plans_with_lm(
                #         exploratory_plans=pruned_plans,
                #         state=initial_state,
                #         world_model_str=self.runtime_vars['world_model_str']
                #     )

                # print(f"LLM Pruned exploratory plans: {LLM_pruned_plans}")
                
                # self.runtime_vars['exploratory_plans'] = LLM_pruned_plans

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

                        if isinstance(self.engine, BabyAI):
                            mission = self.engine.mission

                        if self._do_revise_model(error_count):
                            prompt = self.revise_world_model_prompt.format(
                                actions_set=self.engine.actions_set,
                                errors_from_world_model='\n\n'.join(examples),
                                world_model_str=self.runtime_vars['world_model_str'],
                                utils="directions = {\n    'left': [-1, 0],\n    'right': [1, 0],\n    'up': [0, 1],\n    'down': [0, -1],\n}",
                                mission_goal=mission
                            )

                            print(prompt)

                            # Create step directory and save files
                            step_dir = self.logger.create_step("revision")
                            resp, completion = self.query_lm(prompt)

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
                                    json.dump(
                                        completion,
                                        f,
                                        default=lambda o: getattr(o, "to_dict", lambda: str(o))(),
                                        indent=2
                                    )

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



                        # self.engine.save_screen(self.logger.experiment_dir + 'post_exploration')
                        # self.engine.save_screen()

                        
                    if isinstance(self.engine, BabyAI):
                        mission = self.engine.mission

                    prompt = self.revise_world_model_prompt.format(
                        actions_set=self.engine.actions_set,
                        errors_from_world_model='\n\n'.join(aggregated_examples),
                        world_model_str=self.runtime_vars['world_model_str'],
                        utils="directions = {\n    'left': [-1, 0],\n    'right': [1, 0],\n    'up': [0, 1],\n    'down': [0, -1],\n}",
                        mission_goal=mission

                    )

                    file_name='current_prompt.txt'
                    # Get the content of the first message in the prompt

                    # Create or open the file and write the prompt content to it
                    with open(file_name, 'w') as file:
                        file.write(prompt) 

                    print(prompt)

                    # Create step directory and save files
                    step_dir = self.logger.create_step("revision")
                    resp, completion = self.query_lm(prompt)

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
                            json.dump(
                                completion,
                                f,
                                default=lambda o: getattr(o, "to_dict", lambda: str(o))(),
                                indent=2
                            )
                        # Update world model code and version
                        self.runtime_vars['world_model_str'] = new_world_model_code
                        self.runtime_vars['error_msg_model'] = new_world_model_code

                        # Overwrite the current worldmodel.py file with the new model
                        self.overwrite_world_model(new_world_model_code)

                    self.tape[-1]['revision_prompts'] = prompt
                    self.tape[-1]['revision_responses'] = resp
                    print(prompt)
                    print(resp)
                    # if self._do_revise_plan(error_count):
                    #     self.runtime_vars['revise_plan'] = True

                    revision_count += 1
                    model_was_revised = True


                    if revision_count > max_revisions:
                        print("Max model revisions reached. Exiting.")
                        break
                    print(f"Model revised {revision_count} times. Re-running.")

            else:
                # If no revisions happened and no win occurred, stop the loop
                break

            self._update_solution(self.current_level, first_letters)
            self.level_statistics[f"{self.engine.level_set}_{self.current_level}"]["first_letters"] = first_letters
            self.level_statistics[f"{self.engine.level_set}_{self.current_level}"]["revisions"] = revision_count
            self.level_statistics[f"{self.engine.level_set}_{self.current_level}"]["debugs"] = debug_count
        print("LEVEL TERMINATED")
        print(self.engine.get_obs())
        self._update_solution(self.current_level, first_letters)
        self.level_statistics[f"{self.engine.level_set}_{self.current_level}"]["first_letters"] = first_letters
        self.level_statistics[f"{self.engine.level_set}_{self.current_level}"]["revisions"] = revision_count
        self.level_statistics[f"{self.engine.level_set}_{self.current_level}"]["debugs"] = debug_count

        # At the end of the run, generate and save summary
        summary = f"""
Level: {self.current_level}
Revisions: {revision_count}
Attempts: {attempt_count}
Final Status: {"Won" if self.engine.won else "Failed"}
First Letters: {first_letters}
        """
        self.logger.save_summary(summary)


        # self.engine.save_screen()

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
                    self.engine = BabyAI(level_set=level_set, level_id=level_id)
                elif args.game == 'boulderdash2':
                    self.engine = Boulderdash2Env(level_set=level_set, level_id=level_id)
                elif args.game == 'pb1':
                    self.engine = pb1env(level_set=level_set, level_id=level_id)
                elif args.game == 'sokoban':
                    self.engine = SokobanEnv(level_set=level_set, level_id=level_id)
                elif args.game == 'clusterbox':
                    self.engine = ClusterboxEnv(level_set=level_set, level_id=level_id)
                
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

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, default='pb1')
    parser.add_argument('--level-sets', type=str, default="{'pb1': [0, 1, 2, 3]}")
    parser.add_argument('--episode-length', type=int, default=20)
    parser.add_argument('--world-model-file-name', type=str, default='worldmodel')
    parser.add_argument('--domain-file-name', type=str, default='domain.pddl')  # Add this line
    parser.add_argument('--predicates-file-name', type=str, default='predicates')
    parser.add_argument('--json-reporter-path', type=str, default='KekeCompetition-main/Keke_JS/reports/TBRL_BABA_REPORT.json')
    parser.add_argument('--learn-model', action='store_true')
    parser.add_argument('--query-mode', type=str, default='groq')
    parser.add_argument('--experiment-dir', type=str, default='debuglv3',
                      help='Directory to store experiment runs')
    parser.add_argument('--multi-level', action='store_true', help='Run multiple levels sequentially')
    parser.add_argument('--max-attempts', type=int, default=4, help='Maximum attempts per level')
    parser.add_argument('--prune-plans', action='store_true', help='Use the current method of handling one subplan at a time')
    args = parser.parse_args()

    level_sets = eval(args.level_sets)

    # plan_file_name='sokoban_plans.json'
    # plan_file_name='sokoban_plans2.json'
    plan_file_name='plans.json'
    # plan_file_name='clusterbox_plans.json'


    agent = TheoryCoderAgent(
        base_dir=args.experiment_dir,  # Use experiment directory from args
        episode_length=args.episode_length,
        world_model_load_name=args.world_model_file_name,
        json_reporter_path=args.json_reporter_path,
        predicates_file_name=args.predicates_file_name,
        domain_file_name=args.domain_file_name, 
        do_revise_model=args.learn_model,
        plans_file_name=plan_file_name,  # Pass the specific plan file
        query_mode=args.query_mode,
        prune_plans=args.prune_plans
    )

    if args.multi_level:
        results = agent.run_multiple_levels(level_sets, max_revisions=5, max_attempts=6)
        print("\nExperiment Complete!")
        print(f"Levels Completed: {len(results['levels_completed'])}")
        print(f"Levels Failed: {len(results['levels_failed'])}")
    else:
        if args.game == 'baba':
            engine = BabaIsYou(level_set=list(level_sets.keys())[0], level_id=level_sets[list(level_sets.keys())[0]][0])
        elif args.game == 'lava':
            engine = LavaGrid()
        elif args.game == 'babyai':
            engine = BabyAI(level_set=list(level_sets.keys())[0], level_id=level_sets[list(level_sets.keys())[0]][0])
        elif args.game == 'boulderdash2':
            engine = Boulderdash2Env(level_set=list(level_sets.keys())[0], level_id=level_sets[list(level_sets.keys())[0]][0])
        elif args.game == 'pb1':
            engine = pb1env(level_set=list(level_sets.keys())[0], level_id=level_sets[list(level_sets.keys())[0]][0])
        elif args.game == 'sokoban':
            engine = SokobanEnv(level_set=list(level_sets.keys())[0], level_id=level_sets[list(level_sets.keys())[0]][0])
        elif args.game == 'labyrinth':
            engine = LabyrinthEnv(level_set=list(level_sets.keys())[0], level_id=level_sets[list(level_sets.keys())[0]][0])
        elif args.game == 'cheesemaze':
            engine = CheesemazeEnv(level_set=list(level_sets.keys())[0], level_id=level_sets[list(level_sets.keys())[0]][0])
        elif args.game == 'clusterbox':
            engine = ClusterboxEnv(level_set=list(level_sets.keys())[0], level_id=level_sets[list(level_sets.keys())[0]][0])
        

        agent.run(engine, max_attempts=args.max_attempts)

    # Save tape to json
    import time
    import json
    tape_path = f'tapes/{args.game}_{list(level_sets.keys())[0]}_{level_sets[list(level_sets.keys())[0]][0]}_{time.time()}.json'
    Path(tape_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(tape_path, 'w') as f:
        json.dump(agent.tape, f, indent=4)