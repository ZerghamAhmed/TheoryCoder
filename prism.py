"""
"""
import importlib
from pathlib import Path
from copy import deepcopy
import json
import os
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import random
import re
from games import LavaGrid, BabaIsYou
import ast
from levelrunner import actor
from babareport import BabaReportUpdater
import inspect
from worldmodeltracker import save_world_model
from predicates import rule_formed
import utils
from baselines import *
from preprocessing import *
import openai
from groq import Groq
from experiment_logger import ExperimentLogger
from time import strftime, gmtime




initialize_world_model_prompt = \
"""You are an AI agent that must come up with a transition model of the game you are playing. 

A BFS low-level planner that will use your synthesized transition model to find the low-level actions that will allow you to win levels of the game.

You are also given state transition after executing random actions that will help as well.
Note that if there is no change returned after doing that action, it means that moving was prevented somehow such as by an obstacle. 

The levels you start out with will be simpler but you will be adding on more and more as time progresses. 
So try to make the transition model general and avoid hardcoding anything from the state dictionary keys. Feel free to infer the types of interactions that will occur in later levels. 
Do not feel like you need to build the transition model for just this replay buffer. 
For example, make sure you use each of the categorizations i.e. overlappables, pushables, controllables, etc in your initial world model.
Please make sure to handle pushables in your initial world model too. Pushables cannot overlap with eachother.

BUT A PUSHABLE DOES PUSH ANOTHER PUSHABLE. MAKE NOTE OF THIS if a pushable is pushed and another pushable is occupying that space, that pushable will be pushed
unless there is a border!

Do not assume the win condition is always the same for future levels. Do not change or deal with the win key in the state. 
This will be handled by the game engine, not the transition model.

Do NOT just assume there will be one controllable object, you will need to loop over all the controllables.

Remember there should be logic in your transition model to handle the pushables pushing other pushables!!!!

Also, remember to return the state if you modified it directly or return the new_state if you deepcopied it.

 

CURRENT STATE:

{current_state}

ACTION SPACE:

{actions_set}

UTILS:

{utils}


RESPONSE FORMAT:

- Make sure that you return the correct state for example if you made a deepcopy of the state and modify the deep copy then return the new_state
- If you modify the state directly then return the state instead of new_state

```python

# make sure to include these import statements
from predicates import *
from copy import deepcopy
from utils import directions

def transition_model(state, action):


	Return State

```
"""


revise_world_model_prompt = \
""" You are an AI agent that must come up with a model of the game you are playing. This model you are making of the game
will be a python program that captures the logic and mechanics of the game. You have begun this world model, but it did not capture everything. Below is your current world model, the action space, and the state transition that your transition model handled wrong.

In order to craft the world model and get this state transition you explored your environment with an EXPLORATION PLAN, The state initially and after executing this plan is shown below. Pay close attention to what is involved and modify your transition model to be able to handle this.

Note this exploration is a high level plan and the transition returned is at the end of executing all the actions(right, left, up, down) related to this high level plan.

Notes:

Also DO NOT make changes to "won" in the state dictionary since that will happen outside of the world model.

Feel free to also explain your thinking outside of the markup tags, but know that I will only use the code inside the markup tags. 

Your response format is given below. You will never need to modify any of the current program loops. Your code should only be an addition outside of the current logic just as shown in the example.

You can access the current rules_formed by state['rules_formed'], pushables by state['pushables'], etc.. You DO NOT need to handle their modification, the game engine returns them updated.  

However, based on the rules_formed you may need to adjust the entities or other dictionary elements.
 

ACTION SPACE:

{actions_set}

STATE FORMAT: 

{state_format}

CURRENT WORLD MODEL:

{world_model_str}


ERRORS FROM WORLD MODEL:

{errors_from_world_model}

UTILS:

{utils}


RESPONSE FORMAT (make sure to include your code in markup tags):

```Python

# make sure to include these import statements
from predicates import *
from copy import deepcopy
from utils import directions

def transition_model(state, action):

       if 'x_word y_word z_word' in state['rules_formed']:
              state[x_obj] # example logic
        else:
              state[y_obj] # example logic
	


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

STATE FORMAT: 

{state_format}

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
from games import BabaIsYou
from babareport import BabaReportUpdater
from utils import directions

def transition_model(state, action):


	Return State

```

"""


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

class PRISMAgent:
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
        episode_length=20,
        do_revise_model=False,
        sparse_interactions=True,  # Only run subset of world model
        observation_memory_size=1,
        planner_explore_prob=0,
        max_replans=1,
        plans_file_name='plans.json',  # Default to a generic file if not specified
        base_dir=None,  # Added for experiment logging
        experiment_name=None  # Added for experiment logging
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

        # Initialize version counter
        try:
            with open('world_model_version.txt', 'r') as version_file:
                self.world_model_version = int(version_file.read())
        except FileNotFoundError:
            self.world_model_version = 0


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
            self.llm_client = ChatOpenAI(model_name=language_model, temperature=temperature)
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

        # Set up chat model
        self.language_model = language_model
        self.temperature = temperature
        chat = ChatOpenAI(
            model_name=self.language_model,
            temperature=temperature
        )
        # self.query_lm = lambda prompt: chat(prompt.to_messages()).content
        self.episode_length = episode_length
        self.groq_model = groq_model


        # Record episodes
        self.tape = [{}]

        # Dynamically load plans
        self.plans_file_name = plans_file_name
        self.plans = self._load_plans()

        self.domain_file = 'domain.pddl'
        self.predicates_file_name = 'predicates'

        # Initialize the updater
        self.updater = BabaReportUpdater(json_reporter_path) if json_reporter_path else None

        self.world_model_empty = False  # Flag for empty model
        # self.world_model_available = False  # Default to False


         # Load domain PDDL and predicates files
        self._load_domain_pddl(self.domain_file)
        self._load_predicates(self.predicates_file_name)
        
        self.load_utils()

        # Add new runtime variables to track exploratory plans
        self.runtime_vars['exploratory_plans'] = []
        self.runtime_vars['unsatisfied_preconditions'] = []

        # Initialize experiment logger
        self.logger = ExperimentLogger(base_dir or os.getcwd(), experiment_name)

        # Initialize level statistics
        self.level_statistics = {}

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
            return completion.choices[0].message.content.strip(), completion.system_fingerprint if completion.system_fingerprint else ""

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
            """Load plans from the specified plans file."""
            try:
                with open(self.plans_file_name, 'r') as f:
                    return json.load(f)
            except FileNotFoundError:
                print(f"Plans file '{self.plans_file_name}' not found. Using an empty plan set.")
                return {}  # Return an empty plan set if the file is missing
    

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

        # breakpoint()
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

        all_keys.remove("won")
        # all_keys.remove("empty")

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
        if self.updater:
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
                    state_format=self.engine.state_format,
                    actions_set=self.engine.actions_set,
                    world_model_str=self.runtime_vars['world_model_str'],
                    observations='IGNORE',
                    state=state,
                    action=action,
                    error=e,
                    utils=self.runtime_vars['utils']
                )

                # Use experiment logger to save debug files
                step_dir = self.logger.create_step("debug")
                resp, fingerprint = self.query_lm(prompt)
                new_world_model_code = self.extract_code_from_response(resp)

                if new_world_model_code:
                    self.logger.save_step_files(
                        step_dir,
                        prompt,
                        resp,
                        new_world_model_code,
                        "worldmodel.py"
                    )

                    # Save fingerprint to file
                    with open(os.path.join(step_dir, "fingerprint.txt"), "w") as f:
                        f.write(fingerprint)

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
                    save_world_model("world_model_versions/debugging/", iteration=self.world_model_version)

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

        if self._do_revise_model(error_count):
            prompt = self.revise_world_model_prompt.format(
                state_format=self.engine.state_format,
                actions_set=self.engine.actions_set,
                errors_from_world_model='\n\n'.join(examples),
                world_model_str=self.runtime_vars['world_model_str'],
                utils=self.runtime_vars['utils']
            )

            # Create step directory and save files
            step_dir = self.logger.create_step("revision")
            resp, fingerprint = self.query_lm(prompt)
            new_world_model_code = self.extract_code_from_response(resp)

            if new_world_model_code:
                self.logger.save_step_files(
                    step_dir,
                    prompt,
                    resp,
                    new_world_model_code,
                    "worldmodel.py"
                )

                # Save fingerprint to file
                with open(os.path.join(step_dir, "fingerprint.txt"), "w") as f:
                    f.write(fingerprint)

                self.logger.add_to_tape({
                    "step": "revision",
                    "prompt": prompt,
                    "response": resp
                })

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
        utils="directions = {\n    'left': [-1, 0],\n    'right': [1, 0],\n    'up': [0, 1],\n    'down': [0, -1],\n}"
    )

        file_name='current_prompt.txt'
        # Get the content of the first message in the prompt
        prompt_content = prompt

        # breakpoint()

        # Create or open the file and write the prompt content to it
        with open(file_name, 'w') as file:
            file.write(prompt_content)        
        resp, fingerprint = self.query_lm(prompt)

        new_world_model_code = self.extract_code_from_response(resp)

        if new_world_model_code:
            # Update the world model string in runtime_vars
            self.runtime_vars['world_model_str'] = new_world_model_code

            # Overwrite the current worldmodel.py file with the new model
            self.overwrite_world_model(new_world_model_code)

            # Save the new model as an iteration (e.g., _iteration1)
            save_world_model("world_model_versions/", iteration=0)

            # Create step directory and save files
            step_dir = self.logger.create_step("initialize")
            self.logger.save_step_files(
                step_dir,
                prompt,
                resp,
                new_world_model_code,
                "worldmodel.py"
            )

            # Save fingerprint to file
            with open(os.path.join(step_dir, "fingerprint.txt"), "w") as f:
                f.write(fingerprint)

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

        if mode == 'explore_collision':

            actions = []
            state = self.engine.get_obs()
            state = {key: [list(item) for item in value] if isinstance(value, list) else value for key, value in state.items()}
            # breakpoint()
            import worldmodel
            import planner
            import levelrunner
            importlib.reload(worldmodel)
            importlib.reload(planner)
            importlib.reload(levelrunner)

            actionlist, state = actor(self.domain_file, subplan_exploratory, state, max_iterations=None, debug_callback=self._call_model_debug, level=self.current_level) #max its 2k original
            # breakpoint()
            actions.extend(actionlist)

            return actions
        
        else:
            for i in range(self.max_replans):
                # Use the actor function to generate the actions
                # breakpoint()
                actions = []
                state = self.engine.get_obs()
                state = {key: [list(item) for item in value] if isinstance(value, list) else value for key, value in state.items()}
                # breakpoint()
                import worldmodel
                import planner
                import levelrunner
                importlib.reload(worldmodel)
                importlib.reload(planner)
                importlib.reload(levelrunner)
                importlib.reload(utils)

                
                for subplan in self.plans.get(str(self.current_level), []):
                    # breakpoint()
                    action_seq, state = actor(self.domain_file, subplan, state, max_iterations=None, debug_callback=self._call_model_debug, level=self.current_level) #max its 2k original
                    actions.extend(action_seq)

            return actions  # Return the actions as a list


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
                save_world_model("world_model_versions/original/", iteration=0)

            except AttributeError:
                print("Warning: transition_model function not found.")
                self.world_model_empty = True
            except Exception as e:
                print(f"Error loading world model: {e}")
                self.world_model_empty = True
        else:
            print(f"World model file '{world_model_load_name}.py' not found.")
            self.world_model_empty = True

    def capture_world_model(self):
        # Path to the worldmodel.py file
        world_model_path = "worldmodel.py"
        
        # Read the entire content of the file
        with open(world_model_path, 'r') as file:
            world_model_str = file.read()
        
        # Store the content in runtime_vars
        self.runtime_vars['world_model_str'] = world_model_str

        save_world_model("world_model_versions/", iteration=0)

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
        with Path(self.predicates_save_name + '.py').open('w') as fid:
            fid.write(self.runtime_vars['predicates'].replace('{{', '{').replace('}}', '}'))


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

    def overwrite_world_model(self, new_code):
        # Path to the current worldmodel.py file
        world_model_path = "worldmodel.py"
        
        # Write the new code to the worldmodel.py file
        with open(world_model_path, 'w') as file:
            file.write(new_code)

    def extract_code_from_response(self, response):
        # Use a regular expression to extract the Python code within ```python ``` tags (case-insensitive)
        code_match = re.search(r'```python(.*?)```', response, re.DOTALL | re.IGNORECASE)
        if code_match:
            return code_match.group(1).strip()
        else:
            return None


    def reset(self, keep_model=True):
        self.engine.reset()
        self.runtime_vars['revise_plan'] = False
        self.actions_set = self.engine.actions_set

        state = self.engine.get_obs().copy()
        state = {key: [list(item) for item in value] if isinstance(value, list) else value for key, value in state.items()}

        controllables = {
                    entity for entity in state
                    if rule_formed(state, f'{entity[:-4]}_word', 'is_word', 'you_word')
                }
        
        overlappables = {
                    entity for entity in state
                    if rule_formed(state, f'{entity[:-4]}_word', 'is_word', 'win_word') 
                }
        
        # pushables = {
        #             entity for entity in state
        #             if entity.endswith('_word')
        #         }
        
        pushables = {
        entity for entity in state
            if entity.endswith('_word') 
            or rule_formed(state, f'{entity[:-4]}_word', 'is_word', 'push_word') 
            or (entity.endswith('_obj') and rule_formed(state, f'{entity[:-4]}_word', 'is_word', 'push_word'))
        }

        
        # Add controllables to the state dictionary
        state['controllables'] = list(controllables)

        if 'empty' in state:
            del state['empty']

        if 'won' in state:
            del state['won']

        state['overlappables'] = list(overlappables)

        #  Add controllables to the state dictionary
        state['pushables'] = list(pushables)

        word_entities = [entity for entity in state.keys() if entity.endswith('_word')]
        rules_on_map = []
        for subj in word_entities:
            for pred in word_entities:
                for obj in word_entities:
                    if rule_formed(state, subj, pred, obj):
                        # print(f"Rule formed: {subj} {pred} {obj}")
                        rules_on_map.append(subj + ' ' + pred + ' ' + obj)


        state['rules_formed'] = rules_on_map

        # breakpoint()
        
        self.runtime_vars['observations'] = [state]
        self.actions = []
        self.replay_buffers = []

        # self._generate_rule_stubs() if theorybased
        self.capture_world_model()
        
        # Check if the world model is empty
        if self.is_world_model_empty():
            print("Detected an empty world model.")
            # breakpoint()
            # self._initialize_world_model()
            # breakpoint()  # Trigger a breakpoint if the model is effectively empty

        if self.predicates_empty:
            print("Warning: Predicates file is empty or contains no valid functions/classes.")
            breakpoint()
    
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

        # breakpoint()

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

    
    def enumerate_possible_subplans(self, state):
        """
        Enumerate all possible subplans based on the current state by grounding operators with entities.
        Includes:
        - form_rule for word entities.
        - move_to for object entities.
        - break_rule for existing rules.

        Args:
            state (dict): The current game state.

        Returns:
            list: A list of possible subplans.
        """
        subplans = []

        # Extract entity types from state
        words = [key for key in state if key.endswith('_word')]
        objects = [key for key in state if key.endswith('_obj')]

        # 1. Generate subplans for form_rule (only applicable to word entities)
        for word1 in words:
            for word2 in words:
                for word3 in words:
                    if word1 != word2 and word2 != word3 and word1 != word3:
                        subplans.append(f"form_rule {word1} {word2} {word3}")
        
        for word1 in words:
            for word2 in words:
                for word3 in words:
                    if word1 != word2 and word2 != word3:
                        subplans.append(f"break_rule {word1} {word2} {word3}")

        # 2. Generate subplans for move_to (only applicable to object entities)
        for obj1 in objects:
            for obj2 in objects:
                if obj1 != obj2:  # Avoid self-referencing moves
                    subplans.append(f"move_to {obj1} {obj2}")

        # Generate subplans for push_to
        for pusher in objects:
            if not rule_formed(state, f"{pusher[:-4]}_word", "is_word", "you_word"):  # Ensure pusher is controllable
                continue  # Skip objects that are not pushers

            for obj in objects:
                if not rule_formed(state, f"{obj[:-4]}_word", "is_word", "push_word"):  # Ensure object is pushable
                    continue  # Skip objects that are not pushable

                for target in objects:  # Potential targets
                    if target == obj or target == pusher:  # Avoid self-referencing or invalid targets
                        continue

                    # Add the subplan without indices
                    subplans.append(f"push_to {pusher} {obj} {target}")

        return subplans

    
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

        # Enumerate possible subplans
        possible_subplans = self.enumerate_possible_subplans(state)
        # breakpoint()

        # Load operators from domain file
        for subplan in possible_subplans:
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
        state = {key: [list(item) for item in value] if isinstance(value, list) else value for key, value in state.items()}

        controllables = {
                    entity for entity in state
                    if rule_formed(state, f'{entity[:-4]}_word', 'is_word', 'you_word')
                }
        
        overlappables = {
                    entity for entity in state
                    if rule_formed(state, f'{entity[:-4]}_word', 'is_word', 'win_word')

                }
        
        # pushables = {
        #         entity for entity in state
        #         if entity.endswith('_word')
        #     }
        
        pushables = {
        entity for entity in state
            if entity.endswith('_word') 
            or rule_formed(state, f'{entity[:-4]}_word', 'is_word', 'push_word') 
            or (entity.endswith('_obj') and rule_formed(state, f'{entity[:-4]}_word', 'is_word', 'push_word'))
        }
        
    
        
        # Add controllables to the state dictionary
        state['controllables'] = list(controllables)
        state['overlappables'] = list(overlappables)
        state['pushables'] = list(pushables)

        if 'empty' in state:
            del state['empty']

        word_entities = [entity for entity in state.keys() if entity.endswith('_word')]
        rules_on_map = []
        for subj in word_entities:
            for pred in word_entities:
                for obj in word_entities:
                    if rule_formed(state, subj, pred, obj):
                        # print(f"Rule formed: {subj} {pred} {obj}")
                        rules_on_map.append(subj + ' ' + pred + ' ' + obj)


        state['rules_formed'] = rules_on_map



        # breakpoint()

        self.runtime_vars['observations'].append(state)
        self.actions.append(action)
        # self._make_observation_summaries()  # Formatted for LLM prompts

        # Update replay buffers
        self._update_replay_buffers((
            self.runtime_vars['observations'][-2],
            self.actions[-1],
            self.runtime_vars['observations'][-1]
        ))

        # breakpoint()

        self.tape[-1]['action'] = action
        self.tape[-1]['observation'] = deepcopy(self.runtime_vars['observations'][-1])
        self.tape[-1]['world_model'] = self.runtime_vars['interaction_rules_str']


    def run(self, engine, max_revisions=10, max_attempts=3):
        self.engine = engine
        self.current_level = self.engine.level_id  # Or any other method to determine the level
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
            initial_state = {key: [list(item) for item in value] if isinstance(value, list) else value for key, value in initial_state.items()}

            if self.is_world_model_empty() and self.do_revise_model:
                # If the world model is empty, use the default action set
                print("World model is empty, executing random actions.")
                num_actions = 5
                plan = self.execute_random_actions(num_actions=num_actions)  # Adjust the number as needed
                print(plan)
                print("World model was empty, revised the model. Moving to next iteration.")
                for action in plan:
                    self.step_env(action)
                self._initialize_world_model(num_actions)
                # If the world model is not empty, proceed with the hierarchical planner
                mode = self._sample_planner_mode()  # Determine planner mode (explore/exploit)
                plan = self._hierarchical_planner(mode) 
                print("plan from init model:", plan)
            else:
                # If the world model is not empty, proceed with the hierarchical planner
                mode = self._sample_planner_mode()  # Determine planner mode (explore/exploit)
                plan = self._hierarchical_planner(mode) 
    
            for action in plan:
                self.step_env(action)
                first_letters += action[0]  # Collect the first letters of each action

                # Exit if agent won
                if self.engine.won:
                    self.tape[-1]['exit_condition'] = 'won'
                    self._update_solution(self.current_level, first_letters)
                    self.level_statistics[f"{self.engine.level_set}_{self.current_level}"]["first_letters"] = first_letters
                    self.level_statistics[f"{self.engine.level_set}_{self.current_level}"]["revisions"] = revision_count
                    self.level_statistics[f"{self.engine.level_set}_{self.current_level}"]["debugs"] = debug_count
                    print(first_letters)
                    return True

                # Check if the agent lost (e.g., died or failed critically)
                if self.engine.lost:
                    self.tape[-1]['exit_condition'] = 'lost'
                    self._update_solution(self.current_level, first_letters)
                    self.level_statistics[f"{self.engine.level_set}_{self.current_level}"]["first_letters"] = first_letters
                    self.level_statistics[f"{self.engine.level_set}_{self.current_level}"]["revisions"] = revision_count
                    self.level_statistics[f"{self.engine.level_set}_{self.current_level}"]["debugs"] = debug_count
                    print("AGENT LOST")
                    self._revise_world_model()
                    attempt_count += 1
                    model_was_revised = True
                    break

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
                        self._update_solution(self.current_level, first_letters)
                        self.level_statistics[f"{self.engine.level_set}_{self.current_level}"]["first_letters"] = first_letters
                        self.level_statistics[f"{self.engine.level_set}_{self.current_level}"]["revisions"] = revision_count
                        self.level_statistics[f"{self.engine.level_set}_{self.current_level}"]["debugs"] = debug_count
                        print(first_letters)
                        return True

                    # Check if the agent lost (e.g., died or failed critically)
                    if self.engine.lost:
                        self.tape[-1]['exit_condition'] = 'lost'
                        self._update_solution(self.current_level, first_letters)
                        self.level_statistics[f"{self.engine.level_set}_{self.current_level}"]["first_letters"] = first_letters
                        self.level_statistics[f"{self.engine.level_set}_{self.current_level}"]["revisions"] = revision_count
                        self.level_statistics[f"{self.engine.level_set}_{self.current_level}"]["debugs"] = debug_count
                        print("AGENT LOST")
                        attempt_count += 1
                        break

            # Handle model revision if necessary
            if not self.is_world_model_empty() and self.do_revise_model and not model_was_revised:
                exploratory_plans = self.propose_exploratory_plans(initial_state, self.domain_file)
                print(exploratory_plans)
                # breakpoint()
                pruned_plans = self.prune_exploratory_plans(exploratory_plans)
                print("pruned automatically", pruned_plans)

                # breakpoint()

                if len(pruned_plans) <= 3:
                    print("Executing all pruned plans")
                    LLM_pruned_plans = pruned_plans
                else:
                    # Prune the exploratory plans using LLM
                    LLM_pruned_plans = self.prune_exploratory_plans_with_lm(
                        exploratory_plans=pruned_plans,
                        state=initial_state,
                        world_model_str=self.runtime_vars['world_model_str']
                    )

                print(f"LLM Pruned exploratory plans: {LLM_pruned_plans}")
                
                self.runtime_vars['exploratory_plans'] = LLM_pruned_plans

                # Cycle through exploratory plans across attempts
                for i in range(len(LLM_pruned_plans)):
                    subplan = LLM_pruned_plans[(exploratory_plan_index + i) % len(LLM_pruned_plans)]
                    self.reset(keep_model=True)
                    first_letters = ''  # Reset first_letters for each exploratory plan

                    plan = self._hierarchical_planner(mode="explore_collision", subplan_exploratory=subplan)

                    # use this plan to get D for model revision 
                    if plan == None or plan == ["no-op"]:
                        print(f"plan was none for {subplan}")
                    
                    else:
                        for action in plan:
                            self.step_env(action)
                            first_letters += action[0]  # Collect the first letters of each action

                    # Perform model revision after every collision attempt
                    print(f"Revising the model after subplan: {subplan}")
                    examples, error_count = self._choose_synthesis_examples(exploratory_plan=subplan)
                    if self._do_revise_model(error_count):
                        prompt = self.revise_world_model_prompt.format(
                            state_format=self.engine.state_format,
                            actions_set=self.engine.actions_set,
                            errors_from_world_model='\n\n'.join(examples),
                            world_model_str=self.runtime_vars['world_model_str'],
                            utils=self.runtime_vars['utils']
                        )

                        # Create step directory and save files
                        step_dir = self.logger.create_step("revision")
                        resp, fingerprint = self.query_lm(prompt)

                        with open('fingerprint_seed_v2_revision_lv4.txt', 'w') as file:
                            file.write(fingerprint)

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

                            # Save fingerprint to file
                            with open(os.path.join(step_dir, "fingerprint.txt"), "w") as f:
                                f.write(fingerprint)

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
                # If no revisions happened and no win occurred, stop the loop
                break

            # Special case for level 6
            if self.current_level == 6 and first_letters == 'rrruuu':
                breakpoint()
                self.tape[-1]['exit_condition'] = 'won'
                self._update_solution(self.current_level, first_letters)
                self.level_statistics[f"{self.engine.level_set}_{self.current_level}"]["first_letters"] = first_letters
                self.level_statistics[f"{self.engine.level_set}_{self.current_level}"]["revisions"] = revision_count
                self.level_statistics[f"{self.engine.level_set}_{self.current_level}"]["debugs"] = debug_count
                print(first_letters)
                return True

            self._update_solution(self.current_level, first_letters)
            self.level_statistics[f"{self.engine.level_set}_{self.current_level}"]["first_letters"] = first_letters
            self.level_statistics[f"{self.engine.level_set}_{self.current_level}"]["revisions"] = revision_count
            self.level_statistics[f"{self.engine.level_set}_{self.current_level}"]["debugs"] = debug_count
        print("FAILED LEVEL")
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

        return False

    def run_multiple_levels(self, level_sets, max_revisions=10, max_attempts=3):
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
    parser.add_argument('--game', type=str, default='baba')
    parser.add_argument('--level-sets', type=str, default="{'demo_LEVELS': [0, 1, 2, 3]}")
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
    parser.add_argument('--max-attempts', type=int, default=3, help='Maximum attempts per level')
    args = parser.parse_args()

    level_sets = eval(args.level_sets)

    plan_file_name = 'plans_demo_LEVELS.json'  # Use a single plan file

    agent = PRISMAgent(
        base_dir=args.experiment_dir,  # Use experiment directory from args
        episode_length=args.episode_length,
        world_model_load_name=args.world_model_file_name,
        json_reporter_path=args.json_reporter_path,
        predicates_file_name=args.predicates_file_name,
        domain_file_name=args.domain_file_name, 
        do_revise_model=args.learn_model,
        plans_file_name=plan_file_name,  # Pass the specific plan file
        query_mode=args.query_mode
    )

    if args.multi_level:
        results = agent.run_multiple_levels(level_sets, max_revisions=10, max_attempts=args.max_attempts)
        print("\nExperiment Complete!")
        print(f"Levels Completed: {len(results['levels_completed'])}")
        print(f"Levels Failed: {len(results['levels_failed'])}")
    else:
        if args.game == 'baba':
            engine = BabaIsYou(level_set=list(level_sets.keys())[0], level_id=level_sets[list(level_sets.keys())[0]][0])
        elif args.game == 'lava':
            engine = LavaGrid()
        agent.run(engine, max_attempts=args.max_attempts)

    # Save tape to json
    import time
    import json
    tape_path = f'tapes/{args.game}_{list(level_sets.keys())[0]}_{level_sets[list(level_sets.keys())[0]][0]}_{time.time()}.json'
    Path(tape_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(tape_path, 'w') as f:
        json.dump(agent.tape, f, indent=4)
