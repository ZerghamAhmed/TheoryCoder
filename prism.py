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




initialize_world_model_prompt = \
"""You are an AI agent that must come up with a transition model of the game you are playing. 

A BFS low-level planner that will use your synthesized transition model to find the low-level actions that will allow you to win levels of the game.

You are also given state transition after executing random actions that will help as well.
Note that if there is no change returned after doing that action, it means that moving was prevented somehow such as by an obstacle. 

The levels you start out with will be simpler but you will be adding on more and more as time progresses. 
So try to make the transition model general and avoid hardcoding anything from the state dictionary keys. Feel free to infer the types of interactions that will occur in later levels. 
Do not feel like you need to build the transition model for just this replay buffer. 
For example, make sure you use each of the categorizations i.e. overlappables, pushables, controllables, etc in your initial world model.

Do not assume the win condition is always the same for future levels.
 

CURRENT STATE:

{current_state}

ACTION SPACE:

{actions_set}

Replay Buffer (last {num_random_actions} transitions):

{errors_from_world_model}

UTILS:

{utils}


RESPONSE FORMAT:

```python

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


revise_world_model_prompt = \
""" You are an AI agent that must come up with a model of the game you are playing. This model you are making of the game
will be a python program that captures the logic and mechanics of the game. You have began this world model, but it get some 
of the state transitions wrong. Below is your current world model, the action space, and 
the state transitions that you got correct and the ones that you got incorrect.
For the state transitions that are wrong, you will also be provided with that the end state should be after the action. 
You will also be given utilities typically functions or variables you can use in the world model.

If there are any execution errors you will be given those to fix as well.

Please fix your world model to make it work for all the cases and make it be able to return the correct state for the transition. 

Try to make your world model as general as possible and account for possible cases that may arise in the future!

Notes:

Also DO NOT make changes to "won" in the state dictionary since that will happen outside of the world model.

Feel free to also explain your thinking outside of the markup tags, but know that I will only use the code inside the markup tags. 

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
from games import BabaIsYou
from babareport import BabaReportUpdater
from utils import directions

def transition_model(state, action):


	Return State

```
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

def update_tracking_data(current_entities, current_collisions):
        data = load_tracking_data('tracking_data.json')
        
        # Update observed entities
        new_entities = set(current_entities) - set(data['observed_entities'])
        data['observed_entities'].extend(new_entities)

        # Update observed collisions
        new_collisions = set(current_collisions) - set(data['observed_collisions'])
        data['observed_collisions'].extend(new_collisions)
        
        # Save the updated data back to the file
        save_tracking_data('tracking_data.json', data)

def load_tracking_data(file_path):
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
        except FileNotFoundError:
            data = {"observed_entities": [], "observed_collisions": []}
        return data

def save_tracking_data(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


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


        # Load global collisions from file, hardcoded it for now but make it cmd flag
        self.global_collision_file = 'global_collisions.json'
        self.global_collisions = self.load_global_collisions(self.global_collision_file)

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
            messages = [{"role": "user", "content": prompt.to_string()}]

            # Use the OpenAI client to send the request
            completion = self.llm_client.chat.completions.create(
                model=self.language_model,
                messages=messages,
                temperature=self.temperature,
                seed=42,  # Add the seed for reproducibility
            )
            # Return the generated response
            return completion.choices[0].message.content.strip()

        elif self.query_mode == 'groq':
            # Query using Groq API
            messages = [{"role": "user", "content": prompt.to_string()}]  # Format for Groq
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

    def _call_model_debug_TB(self, rule_key, state, action, max_retries=5):
        for i in range(max_retries):
            try:
                preds = self.runtime_vars['interaction_rules'][rule_key].forward(state, action)
                return preds
            except Exception as e:
                # from ipdb import set_trace; set_trace()
                prompt = self._make_langchain_prompt(
                    self.infer_interaction_rule_prompt + self.debug_model_prompt,
                    **{
                        'state_format': self.engine.state_format,
                        'actions_set': self.engine.actions_set,
                        'operators': self.runtime_vars['operators'].replace('{', '{{').replace('}', '}}'),
                        'predicates': self.runtime_vars['predicates'].replace('{', '{{').replace('}', '}}'),
                        'interaction_rules': '',  # TODO: Insert other rules into context?
                        'interaction_rule': self.runtime_vars['interaction_rules_str'][rule_key],
                        'observations': 'IGNORE',
                        'state': state,
                        'action': action,
                        'error': e,
                        'utils': self.runtime_vars['utils']
                    }
                )
                print(f'DEBUG ITER {i}')
                print(f'ERROR: {e}')
                # resp = self.query_lm(prompt)
                resp = 'hi'
                self.tape[-1]['debug_model_prompt'] = prompt.to_messages()[0].content
                self.tape[-1]['debug_model_response'] = resp
                print(resp)
                self.runtime_vars['error_msg_model'] = self._update_world_model(resp, list(self.runtime_vars['interaction_rules'].keys()))
        return
    
    def _call_model_debug(self, state, action, max_retries=3):
        # import worldmodel
        # importlib.reload(worldmodel)

        if not self.do_revise_model:
            return

        for i in range(max_retries):
            try:
                import worldmodel
                importlib.reload(worldmodel)
                pred = worldmodel.transition_model(state, action)
                return pred
            except Exception as e:
                # from ipdb import set_trace; set_trace()
                # breakpoint()
                prompt = self._make_langchain_prompt(
                    self.debug_model_prompt,
                    **{
                        'state_format': self.engine.state_format,
                        'actions_set': self.engine.actions_set,
                        'world_model_str': self.runtime_vars['world_model_str'],
                        'observations': 'IGNORE',
                        'state': state,
                        'action': action,
                        'error': e,
                        'utils': self.runtime_vars['utils']
                    }
                )
                print(f'DEBUG ITER {i}')
                print(f'ERROR: {e}')
                # breakpoint()

                # Save the prompt to a text file
                with open(f'prompts_saved/revision_prompt_{self.world_model_version}.txt', 'w') as file:
                    file.write(prompt.to_messages()[0].content)

                resp = self.query_lm(prompt)
                # resp = 'hi'
                self.tape[-1]['debug_model_prompt'] = prompt.to_messages()[0].content
                self.tape[-1]['debug_model_response'] = resp
                print(resp)


                # Create directories if they don't exist
                base_dir = "world_model_versions"
                full_responses_dir = os.path.join(base_dir, "full_responses")
                os.makedirs(full_responses_dir, exist_ok=True)

                # Track the world model version iteration using string formatting
                version_str = f"version_{self.world_model_version}"

                # Save the full response to a text file (for logging and reference)
                full_response_path = os.path.join(full_responses_dir, f"full_response_{version_str}.txt")
                with open(full_response_path, 'w') as file:
                    file.write(resp)

                new_world_model_code = self.extract_code_from_response(resp)

                if new_world_model_code:
                    # Update the world model string in runtime_vars
                    self.runtime_vars['world_model_str'] = new_world_model_code

                    # perhaps save history of all the error debugs
                    self.runtime_vars['error_msg_model'] = new_world_model_code

                    # Overwrite the current worldmodel.py file with the new model
                    self.overwrite_world_model(new_world_model_code)

                    # Save the new model as an iteration (e.g., _iteration1)
                    self.world_model_version += 1
                    save_world_model("world_model_versions/debugging/", iteration=self.world_model_version)

        return

    def _update_world_model(self, text, rule_keys, save_to_file=True):
        x = re.findall(r'```python([\s\S]*?)```', text)
        if not len(x):
            return 'Exception: No code found'
        x = '\n'.join([xi.strip() for xi in x])
        if x:
            interaction_rules = []  # Gets appended to in LLM-generated code
            try:
                exec(x, globals(), my_locals := {})
            except Exception as e:
                return e
            else:
                if not 'interaction_rules' in my_locals:
                    return 'Exception: Could not find interaction_rules'
                interaction_rules = my_locals['interaction_rules']
                if len(interaction_rules):
                    err_msg = ''
                    for rule in interaction_rules:
                        key = (rule.entity_key1, rule.entity_key2)
                        if key in rule_keys:
                            self.runtime_vars['interaction_rules'][key] = rule
                            s = extract_function_or_class_str(x, rule.__class__.__name__)
                            s += f"\n\ninteraction_rules.append({rule.__class__.__name__}('{key[0]}', '{key[1]}'))"
                            self.runtime_vars['interaction_rules_str'][key] = s
                        else:
                            err_msg += f'Exception: Could not find {key} in rule_keys.\n'
                    # self._save_interaction_rules_to_file()
                    return None if not err_msg else err_msg
                else:
                    return 'Exception: Could not find any interaction rules'
        else:
            return 'Exception: No code found inside Python tags.'

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
            f"Summary of changes:\n{self._get_state_deltas_str(s0, s1)}"
            f"Your prediction errors:\n{errors}\n"
        )


    def _choose_synthesis_examples(self):
        """
        Choose (s0, a) --> s1 transitions from replay buffer as program
        synthesis examples.

        TODO: Make this fancier (e.g. choose most important examples somehow)
        """
        # Simple solution: Just take the last k from the buffer
        # obs = self.replay_buffers[::-1][:5]
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
        
        # breakpoint()
        return examples, error_count

    
    def deep_sort_list(self, value):
        """Recursively sort lists within the dictionary values."""
        if isinstance(value, list):
            return sorted(self.deep_sort_list(v) for v in value)
        return value

    def compare_states(self, state1, state2):
        """
        Compares two state dictionaries considering unordered lists.
        """
        # First, check if the keys match
        if state1.keys() != state2.keys():
            return False
        
        for key in state1:
            val1 = self.deep_sort_list(state1[key])
            val2 = self.deep_sort_list(state2[key])

            if val1 != val2:
                return False
        
        return True
    
    # Function to simulate state transitions based on planner's actions
    def simulate_planner_transitions(self, s0, actions):
        import worldmodel
        importlib.reload(worldmodel)

        predicted_transitions = []
        
        for action in actions:
            s1 = worldmodel.transition_model(deepcopy(s0), action)  
            predicted_transitions.append((s0, action, s1))
            s0 = s1
        
        return predicted_transitions
    
    def compare_transitions(self, replay_buffer, simulated_transitions):
        correct_predictions = []
        incorrect_predictions = []

        for i, (replay_transition, predicted_transition) in enumerate(zip(replay_buffer, simulated_transitions)):
            replay_state, replay_action, replay_next_state = replay_transition
            pred_state, pred_action, pred_next_state = predicted_transition

            if replay_action != pred_action:
                print(f"Action mismatch at step {i}: replay {replay_action} vs planner {pred_action}")
                continue

            if self.compare_states(replay_next_state, pred_next_state):
                correct_predictions.append((i, replay_transition))
            else:
                incorrect_predictions.append((i, replay_transition, predicted_transition, replay_next_state))

        return correct_predictions, incorrect_predictions


    def _choose_synthesis_examples_TB(self, rule_key):
        """
        Choose (s0, a) --> s1 transitions from replay buffer as program
        synthesis examples.

        TODO: Make this fancier (e.g. choose most important examples somehow)
        """
        # Simple solution: Just take last k from buffer
        obs = self.replay_buffers[::-1][rule_key][:self.observation_memory_size]
        preds = [self._call_model_debug(rule_key, s0, a) for (s0, a, s1) in obs]
        errors = [self._get_pred_errors(rule_key, s1, pred) for (s0, a, s1), pred in zip(obs, preds)]
        examples = [self._make_observation_summaries(rule_key, x, e) for x, e in zip(obs, errors)]
        error_count = sum([1 if e else 0 for e in errors])
        return examples, error_count

    def _revise_world_model(self):
        if not self.do_revise_model:
            return

        self.tape[-1]['revision_prompts'] = {}
        self.tape[-1]['revision_responses'] = {}

        examples, error_count = self._choose_synthesis_examples()
        
        if self._do_revise_model(error_count):
            # Modify this to include the collision information
            # collision_entity = self.runtime_vars.get('current_collision', 'No collision entity')
            # collision_info = f"Attempting collision with {collision_entity}"

            # breakpoint()

            prompt = self._make_langchain_prompt(
                self.revise_world_model_prompt,
                **{
                    'state_format': self.engine.state_format,
                    'actions_set': self.engine.actions_set,
                    'errors_from_world_model': '\n\n'.join(examples),
                    'world_model_str': self.runtime_vars['world_model_str'],
                    'utils': self.runtime_vars['utils']
                }
            )

            file_name='revise_prompt_LOSE_GOOP.txt'
        # Get the content of the first message in the prompt
            prompt_content = prompt.to_messages()[0].content

        # Create or open the file and write the prompt content to it
            with open(file_name, 'w') as file:
                file.write(prompt_content)  

            breakpoint()
            # Query the language model for new world model code
            resp = self.query_lm(prompt)
            new_world_model_code = self.extract_code_from_response(resp)

            if new_world_model_code:
                self.runtime_vars['world_model_str'] = new_world_model_code
                self.overwrite_world_model(new_world_model_code)

                # Increment version counter and save the new version
                self.world_model_version += 1
                save_world_model("world_model_versions/", iteration=self.world_model_version)

                # Save full response in case it's needed
                full_response_path = f"world_model_versions/full_responses/response_{self.world_model_version}.txt"
                with open(full_response_path, "w") as f:
                    f.write(resp)

            self.tape[-1]['revision_prompts'] = prompt.to_messages()[0].content
            self.tape[-1]['revision_responses'] = resp
            print(prompt.to_messages()[0].content)
            print(resp)
            breakpoint()
            if self._do_revise_plan(error_count):
                self.runtime_vars['revise_plan'] = True



    def _PRE_revise_world_model(self):
        if not self.do_revise_model:
            return

        self.tape[-1]['revision_prompts'] = {}
        self.tape[-1]['revision_responses'] = {}
        examples, error_count = self._choose_synthesis_examples()
        if self._do_revise_model(error_count):
            prompt = self._make_langchain_prompt(
                self.revise_world_model_prompt,
                **{
                    'state_format': self.engine.state_format,
                    'actions_set': self.engine.actions_set,
                    'errors_from_world_model': '\n\n'.join(examples),
                    'world_model_str': self.runtime_vars['world_model_str'],
                }
            )
            # resp = 'hi'
            resp = self.query_lm(prompt)
            new_world_model_code = self.extract_code_from_response(resp)

            if new_world_model_code:
                self.runtime_vars['world_model_str'] = new_world_model_code
                self.overwrite_world_model(new_world_model_code)

                # Increment version counter and save the new version
                self.world_model_version += 1
                save_world_model("world_model_versions/", iteration=self.world_model_version)

                # Persist the version counter
                with open('world_model_version.txt', 'w') as version_file:
                    version_file.write(str(self.world_model_version))

            self.tape[-1]['revision_prompts'] = prompt.to_messages()[0].content
            self.tape[-1]['revision_responses'] = resp
            print(prompt.to_messages()[0].content)
            print(resp)
            # breakpoint()
            if self._do_revise_plan(error_count):
                self.runtime_vars['revise_plan'] = True



    def _initialize_world_model(self, num_actions):

        examples, error_count = self._choose_synthesis_examples()

        prompt = self._make_langchain_prompt(
            self.initialize_world_model_prompt,  # Initialize World Model prompt template
            **{
                'domain_file': self.runtime_vars.get('domain_file', ''),
                'current_state': self.runtime_vars['observations'][-1],
                'num_random_actions': num_actions,
                'errors_from_world_model': '\n\n'.join(examples),
                'actions_set': self.engine.actions_set,
                'utils':"directions = {\n    'left': [-1, 0],\n    'right': [1, 0],\n    'up': [0, 1],\n    'down': [0, -1],\n}"
            }
        )

        file_name='current_prompt.txt'
        # Get the content of the first message in the prompt
        prompt_content = prompt.to_messages()[0].content

        # Create or open the file and write the prompt content to it
        with open(file_name, 'w') as file:
            file.write(prompt_content)        
        breakpoint()
        resp = self.query_lm(prompt)
        # resp = 'hi'
        # Save the initial version of the world model
        # Extract the Python code from the response

#         resp1 = \
# """ ```python 
# from predicates import *
# from copy import deepcopy
# from games import BabaIsYou

# def transition_model(state, action):    

#     # DO NOT CHANGE THIS!!!!!!!
#     directions = {
#         'left': [-1, 0],
#         'right': [1, 0],
#         'up': [0, 1],
#         'down': [0, -1]
#     }

#     return state
# ```"""
        # breakpoint()
        new_world_model_code = self.extract_code_from_response(resp)

        if new_world_model_code:
            # Update the world model string in runtime_vars
            self.runtime_vars['world_model_str'] = new_world_model_code

            # Overwrite the current worldmodel.py file with the new model
            self.overwrite_world_model(new_world_model_code)

            # Save the new model as an iteration (e.g., _iteration1)
            save_world_model("world_model_versions/", iteration=0)

    def _revise_operators(self):
        # TODO
        pass
        return

    def _revise_predicates(self):
        # TODO
        pass
        return

    def _random_explore(self):
        return [random.choice(self.actions_set)]    

    def _sample_exploratory_goal(self, state, model):
        raise NotImplementedError()
        # prompt = self._make_langchain_prompt(
        #     self.exploratory_goal_prompt,  # TODO
        #     **{
        #         'interaction_rules': '\n'.join(self.runtime_vars['interaction_rules_str'].values()),
        #         'state_format': self.engine.state_format,
        #         'actions_set': self.engine.actions_set,
        #         'current_state': self.runtime_vars['observations'][-1],
        #     }
        # )
        # resp = self.query_lm(prompt)
        # return resp

    def _sample_plan(self, state, goal):
        """
        Generate plan as series of operators
        """
        prompt = self._make_langchain_prompt(
            self.planner_prompt,
            **{
                'state_format': self.engine.state_format,
                'actions_set': self.engine.actions_set,
                'operators': self.runtime_vars['operators'].replace('{', '{{').replace('}', '}}'),
                'predicates': self.runtime_vars['predicates'].replace('{', '{{').replace('}', '}}'),
                'interaction_rules': '\n'.join(self.runtime_vars['interaction_rules_str'].values()),
                'current_state': state,
                'goal': goal,
            }
        )

        resp = 'hi'
        actions, error_msg = self._update_plan(resp)
        if error_msg is None:
            # Record data to tape
            self.tape[-1]['planner_prompt'] = prompt.to_messages()[0].content
            self.tape[-1]['planner_response'] = resp
            print(prompt.to_messages()[0].content)
            print(resp)
        else:
            print('PLANNER ERROR:', error_msg)
            self.tape[-1]['planner_err'] = error_msg
        if not actions:
            actions = self._random_explore()
        return actions

    def _keep_plan(self):
        """Prompt LLM to keep or reject plan it generated."""
        return True  # DEBUG

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

            actionlist, state = actor(self.domain_file, subplan_exploratory, state, max_iterations=None, debug_callback=self._call_model_debug) #max its 2k original
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
                    action_seq, state = actor(self.domain_file, subplan, state, max_iterations=None, debug_callback=self._call_model_debug) #max its 2k original
                    actions.extend(action_seq)

                if self._keep_plan():
                    break
            # breakpoint()
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
        self.actions_set = engine.actions_set

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
        
        pushables = {
                    entity for entity in state
                    if entity.endswith('_word')
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

    def check_collisions_in_replay_buffer(self):
        """
        This method scans the replay buffer for collisions between different entities.
        A collision is defined as two or more entities occupying the same coordinates.

        Returns:
            observed_collisions (set): A set of observed collisions. Each collision is a tuple of the form
                                    ((entity1, entity2), (x, y)), representing the two entities and their collision coordinates.
        """
        observed_collisions = set()
        
        # Extract entity keys (excluding non-colliding types)
        entities = [key for key in self.engine.get_obs().keys() if key not in ['empty', 'border', 'won', 'lost']]

        # Iterate through each experience in the replay buffer
        for experience in self.replay_buffers:
            state, action, next_state = experience

            # Check for collisions in both state and next_state
            for state_to_check in [state, next_state]:
                for i, entity1 in enumerate(entities):
                    for j in range(i + 1, len(entities)):
                        entity2 = entities[j]

                        # Skip comparison if both entities are the same
                        if entity1 == entity2:
                            continue

                        # Get positions of entity1 and entity2
                        positions1 = state_to_check.get(entity1, [])
                        positions2 = state_to_check.get(entity2, [])

                        # Convert all positions to tuples for consistent comparison
                        positions1 = [tuple(pos) for pos in positions1]
                        positions2 = [tuple(pos) for pos in positions2]

                        # Check if any positions overlap
                        for pos1 in positions1:
                            if pos1 in positions2:
                                # collision = ((entity1, entity2), pos1)
                                collision = ((entity1, entity2))
                                observed_collisions.add(collision)
        
        return observed_collisions
    

    def possible_collision_pairs(self):
        """
        Initialize the set of all possible collision pairs based on observed entities.
        """
        entities = [key for key in self.engine.get_obs().keys() if key not in ['empty', 'border', 'won', 'lost']]
        return set((entity1, entity2) for i, entity1 in enumerate(entities) for entity2 in entities[i+1:])
    
    def get_unobserved_collisions(self, observed_collisions):

        all_possible_collisions = self.possible_collision_pairs()  # Call the method to get the set of possible collisions

        # Calculate unobserved collisions by taking the difference
        unobserved_collisions = all_possible_collisions - observed_collisions

        return unobserved_collisions


    def load_global_collisions(self, file_path):
        """
        Load global collision tracking from an external file.
        """
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                global_collisions = json.load(f)
                return set(tuple(collision) for collision in global_collisions)
        else:
            return set()

    def save_global_collisions(self):
        """
        Save the global collision data to an external file.
        """
        with open(self.global_collision_file, "w") as f:
            json.dump([list(collision) for collision in self.global_collisions], f, indent=4)


    def update_global_collisions(self, observed_collisions):
        """
        Update the global collision data with newly observed collisions.
        """
        level_key = f"level_{self.current_level}"

        if level_key not in self.global_collisions:
            self.global_collisions[level_key] = []

        for collision in observed_collisions:
            if collision not in self.global_collisions[level_key]:
                self.global_collisions[level_key].append(collision)

        # Save the updated global collisions to file
        self.save_global_collisions()

    def reorder_collision_pair(self, controllable, entity1, entity2):
        """
        Ensure the controllable entity is first in the collision tuple.

        Args:
            controllable (str): The controllable entity.
            entity1 (str): The first entity in the collision.
            entity2 (str): The second entity in the collision.

        Returns:
            tuple: A tuple with the controllable entity first.
        """
        if entity1 == controllable:
            return (entity1, entity2)
        elif entity2 == controllable:
            return (entity2, entity1)
        else:
            return (entity1, entity2)  # Default order if neither is controllable


    def generate_subplans_for_collisions(self, state, collisions_to_test):
        subplans = []

        for entity1, entity2 in collisions_to_test:
            # Get all instances of entity1 and entity2
            instances1 = state.get(entity1, [])
            instances2 = state.get(entity2, [])

            # Generate subplans for each combination of instances
            for i, instance1 in enumerate(instances1, start=1):
                for j, instance2 in enumerate(instances2, start=1):
                    subplan = f"move_to {entity1}_{i} {entity2}_{j}"
                    subplans.append(subplan)

                    # Save the current collision being tested for use in prompts
                    self.runtime_vars['current_collision'] = entity2

        return subplans

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
            for i, coords1 in enumerate(state[obj1]):
                for obj2 in objects:
                    for j, coords2 in enumerate(state[obj2]):
                        if obj1 != obj2 or i != j:  # Avoid self-referencing moves
                            subplans.append(f"move_to {obj1}_{i+1} {obj2}_{j+1}")

        # Generate subplans for push_to
        for pusher in objects:
            if not rule_formed(state, f"{pusher[:-4]}_word", "is_word", "you_word"):  # Ensure pusher is controllable
                continue  # Skip objects that are not pushers

            for i, coords_pusher in enumerate(state[pusher]):  # Enumerate pusher instances
                for obj in objects:
                    if not rule_formed(state, f"{obj[:-4]}_word", "is_word", "push_word"):  # Ensure object is pushable
                        continue  # Skip objects that are not pushable

                    for j, coords_obj in enumerate(state[obj]):  # Enumerate pushed object instances
                        for target in objects:  # Potential targets
                            if target == obj or target == pusher:  # Avoid self-referencing or invalid targets
                                continue

                            for k, coords_target in enumerate(state[target]):  # Enumerate target instances
                                if coords_pusher == coords_target:  # Avoid pusher overlapping with the target
                                    continue

                                # Add the subplan with indices
                                subplans.append(f"push_to {pusher}_{i+1} {obj}_{j+1} {target}_{k+1}")
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

    def execute_exploratory_plans(self, exploratory_plans, state, domain_file):
        """
        Execute the proposed exploratory plans, accumulate successfully executed plans, and update the state.
        """
        successful_plans = []  # Track successful exploratory plans

        for subplan in exploratory_plans:
            try:
                actions, new_state = actor(domain_file, subplan, state, max_iterations=None)
                print(f"Executed exploratory subplan: {subplan}")
                print(f"Actions: {actions}")
                print(f"Resulting state: {new_state}")

                # Check if actions were successfully executed
                if actions and actions != ["no-op"]:
                    successful_plans.append(subplan)

                # Log the transition to replay buffer
                self._update_replay_buffers((state, subplan, new_state))

                # Update the state
                state = new_state

            except Exception as e:
                print(f"Failed to execute subplan {subplan}: {e}")

        print(f"Successful exploratory plans: {successful_plans}")
        return state, successful_plans


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
        
        pushables = {
                entity for entity in state
                if entity.endswith('_word')
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


    def run(self, engine, max_revisions=1):
        self.engine = engine
        self.current_level = self.engine.level_id  # Or any other method to determine the level
        revision_count = 0

        while revision_count <= max_revisions:
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
                num_actions = 10
                plan = self.execute_random_actions(num_actions=num_actions)  # Adjust the number as needed
                print(plan)
                print("World model was empty, revised the model. Moving to next iteration.")
                for action in plan:
                    self.step_env(action)
                self._initialize_world_model(num_actions)
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
                    print(first_letters)
                    # breakpoint()
                    return True

                # # Check if the agent lost (e.g., died or failed critically)
                # if self.engine.lost:
                #     self.tape[-1]['exit_condition'] = 'lost'
                #     self._update_solution(self.current_level, first_letters)
                #     print("AGENT LOST")
                #     self._revise_world_model()
                #     breakpoint()
                #     return True
                
            # if self.is_world_model_empty() and self.do_revise_model:
            #     revision_count += 1
            #     self._initialize_world_model(num_actions)
            #     print("World model was empty, revised the model. Moving to next iteration.")
            #     continue

            # Handle model revision if necessary
            if not self.is_world_model_empty() and self.do_revise_model:
                # breakpoint()
                exploratory_plans = self.propose_exploratory_plans(initial_state, self.domain_file)
                print(exploratory_plans)
                final_plans = self.prune_exploratory_plans(exploratory_plans)
                print("pruned", final_plans)
                breakpoint()
                if exploratory_plans:
                    print("Generated exploratory plans:", exploratory_plans)
                    state, successful_plans = self.execute_exploratory_plans(exploratory_plans, initial_state, self.domain_file)
                final_plans = self.prune_exploratory_plans(successful_plans)
                self.runtime_vars['exploratory_plans'] = final_plans

                breakpoint()
                subplans = []
                for subplan in subplans:
                    self.reset(keep_model=True)
                    # self.engine.reset()

                    plan = self._hierarchical_planner(mode="explore_collision", subplan_exploratory=subplan)

                    # use this plan to get D for model revision 
                    if plan == None or plan == ["no-op"]:
                        print(f"plan was none for {subplan}")
                        breakpoint()
                    
                    else:
                        breakpoint()
                        for action in plan:

                            self.step_env(action)


                    # Perform model revision after every collision attempt
                    print(f"Revising the model after subplan: {subplan}")
                    self._revise_world_model()

                    revision_count += 1

                    breakpoint()

                    if revision_count > max_revisions:
                        print("Max model revisions reached. Exiting.")
                        break
                    print(f"Model revised {revision_count} times. Re-running.")

                # self._revise_world_model()
                
            else:
                # If no revisions happened and no win occurred, stop the loop
                break

            self._update_solution(self.current_level, first_letters)
        # self._revise_world_model()
        print("FAILED LEVEL")
        self._update_solution(self.current_level, first_letters)
        # print(state)
        return False



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, default='baba')
    parser.add_argument('--levels', type=str, default="[('demo_LEVELS', 1)]")
    parser.add_argument('--episode-length', type=int, default=20)
    parser.add_argument('--world-model-file-name', type=str, default='worldmodel')
    parser.add_argument('--domain-file-name', type=str, default='domain.pddl')  # Add this line
    parser.add_argument('--predicates-file-name', type=str, default='predicates')
    parser.add_argument('--json-reporter-path', type=str, default='KekeCompetition-main/Keke_JS/reports/TBRL_BABA_REPORT.json')
    parser.add_argument('--learn-model', action='store_true')
    parser.add_argument('--query-mode', type=str, default='groq')
    args = parser.parse_args()

    levels = eval(args.levels)

    for level_set, level_id in levels:
        plan_file_name = f'plans_{level_set}.json'  # Dynamically set plan file based on level_set

        if args.game == 'baba':
            engine = BabaIsYou(level_set=level_set, level_id=level_id)
        elif args.game == 'lava':
            engine = LavaGrid()
        agent = PRISMAgent(
            episode_length=args.episode_length,
            world_model_load_name=args.world_model_file_name,
            json_reporter_path=args.json_reporter_path,
            predicates_file_name=args.predicates_file_name,
            domain_file_name=args.domain_file_name, 
            do_revise_model=args.learn_model,
            plans_file_name=plan_file_name,  # Pass the specific plan file
            query_mode=args.query_mode
        )
        agent.run(engine)

    # Save tape to json
    import time
    import json
    tape_path = f'tapes/{args.game}_{level_set}_{level_id}_{time.time()}.json'
    Path(tape_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(tape_path, 'w') as f:
        json.dump(agent.tape, f, indent=4)
