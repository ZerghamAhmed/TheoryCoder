import json
from pathlib import Path
from copy import deepcopy
import os
from langchain.prompts import HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI
from games import BabaIsYou
from predicates import rule_formed
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import re
import ast 
from groq import Groq

# Initialize the Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

INITIAL_REQUEST_PROMPT = """
You are an AI agent that must come up with a list of actions that need to be taken to win a certain level in a game. 
These actions can only come from the action space given below. You are given an example of what your response 
format for this list of actions should look like. You will also need to provide your reasoning for why this will allow you to win.

You are given your current state that you start from in the level. 

So using the information please return the action sequence that will result in winning the level. 
Make sure to give your explanation, also
make sure to just have a sepearte section with your actions as demonstrated in the response format.

ACTION SPACE (YOUR LIST SHOULD BE COMPOSED OF THESE ACTIONS):

{actions_set}

STATE FORMAT:

{state_format}

INITIAL STATE:

{initial_state}

UTILS:

{utils}

RESPONSE FORMAT (just a random example list, make sure your answer is returned with markup tag):

```Python

["right", "left", "up", "down"]

```

explanation:

Example explanation.

"""

REFINE_PROMPT = """
You are an AI agent that must come up with a list of actions that need to be taken to win a certain level in a game. 
These actions can only come from the action space given below. You are given an example of what your response 
format for this list of actions should look like. 

You are given your current state that you start from in the level. 

You previously attemped this level and returned the following action sequences but did not win the game. 
The history of your previous action sequence predictions and the corresponding replay buffer for that sequence is given 
under history.

Please provide your corrected action sequence that will result in winning the level. 
Do not forget to give your explanation for why this is now correct. 

ACTION SPACE (YOUR LIST SHOULD BE COMPOSED OF THESE ACTIONS):

{actions_set}

STATE FORMAT:

{state_format}

INITIAL STATE FOR LEVEL:

{initial_state}

HISTORY:

{action_replay_tuples}


UTILS:

{utils}

RESPONSE FORMAT (just a random example list, make sure your answer is returned with markup tag, explanations should be outside it):

```Python

["right", "left", "up", "down"]

```

explanation:

Example explanation.

"""
import json
from pathlib import Path
from copy import deepcopy
import os
from langchain.prompts import HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
import re
import ast

# Assume that BabaIsYou and rule_formed are defined in their respective modules
from games import BabaIsYou
from predicates import rule_formed

import json
from pathlib import Path
from copy import deepcopy
import os
from langchain.prompts import HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI
from games import BabaIsYou
from predicates import rule_formed
from langchain.prompts.chat import ChatPromptTemplate
import re
import ast

import json
from pathlib import Path
from copy import deepcopy
import os
from langchain.prompts import HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI
from games import BabaIsYou
from predicates import rule_formed
from langchain.prompts.chat import ChatPromptTemplate
import re
import ast
from groq import Groq

class Baselines:
    def __init__(self, episode_length, world_model_load_name, domain_file_name, predicates_file_name,
                 refine=False, max_refinements=5, save_dir="experiment_results", plans_json_path="plans.json", groq=None):
        self.episode_length = episode_length
        self.world_model_load_name = world_model_load_name
        self.domain_file_name = domain_file_name
        self.predicates_file_name = predicates_file_name
        self.refine_enabled = refine
        self.max_refinements = max_refinements
        self.save_dir = Path(save_dir).resolve()
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.language_model = 'vj'  # Update as needed
        self.chat = ChatOpenAI(model_name=self.language_model, temperature=1.0)
        self.query_lm = lambda prompt: self.chat.invoke(prompt.to_messages()).content

        self.tape = []
        self.replay_buffers = []
        self.actions_set = ["up", "down", "left", "right"]
        self.utils = {
            'directions': {
                'left': [-1, 0],
                'right': [1, 0],
                'up': [0, 1],
                'down': [0, -1],
            }
        }
        self.actions = []
        self.initial_state = None
        self.engine = None

        self.plans = self.load_plans(plans_json_path)
        self.groq = groq  

    def load_plans(self, plans_json_path):
        """Load the plans from the specified JSON file."""
        try:
            with open(plans_json_path, 'r') as f:
                plans = json.load(f)
            print(f"Loaded plans from {plans_json_path}")
            return plans
        except FileNotFoundError:
            print(f"Plans JSON file not found at {plans_json_path}. Proceeding without plans.")
            return {}
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {plans_json_path}: {e}. Proceeding without plans.")
            return {}

    def get_plan_for_level(self, level_id):
        """Retrieve the plan for the specified level."""
        level_key = str(level_id)
        plan = self.plans.get(level_key, [])
        print(f"Retrieved plan for level {level_id}: {plan}")
        return plan

    def _make_langchain_prompt(self, text, **kwargs):
        """Create the Langchain prompt with given template and variables."""
        human_template = HumanMessagePromptTemplate.from_template(text)
        chat_prompt = ChatPromptTemplate.from_messages([human_template])
        prompt = chat_prompt.format_prompt(**kwargs)
        return prompt
    
    def groq_query(self, prompt_content):
        """
        Custom querying logic for Groq.
        """
        print("Using Groq prompt:")
        print(prompt_content)
        messages = [{"role": "user", "content": prompt_content}]
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="llama3-8b-8192"
        )
        response = chat_completion.choices[0].message.content.strip()
        return response

    def load_file(self, file_path):
        """Load the contents of a file."""
        with open(file_path, 'r') as f:
            return f.read().strip()

    def format_state(self, state):
        """Convert state tuples to lists and add controllables."""
        state = {key: [list(item) for item in value] if isinstance(value, list) else value for key, value in state.items()}
        controllables = {
            entity for entity in state
            if rule_formed(state, f'{entity[:-4]}_word', 'is_word', 'you_word')
        }
        state['controllables'] = list(controllables)
        return state

    def save_file(self, level_id, status, step, file_type, content):
        """
        Save content to a file with a specific naming convention.

        Parameters:
        - level_id (int): Identifier for the level.
        - status (str): 'initial', 'won', or 'lost'.
        - step (int): Refinement step number. Use 0 for initial guess.
        - file_type (str): 'prompt', 'actions', or 'response'.
        - content (str): Content to save in the file.
        """
        level_dir = self.save_dir / f"level_{level_id}"
        level_dir.mkdir(parents=True, exist_ok=True)

        if status == "initial":
            sub_dir = "initial"
            if file_type == "prompt":
                filename = "prompt.txt"
            elif file_type == "actions":
                filename = "actions.txt"
            elif file_type == "response":
                filename = "response.txt"
            else:
                raise ValueError("Invalid file_type. Must be 'prompt', 'actions', or 'response'.")
            sub_dir_path = level_dir / sub_dir
        elif status == "won":
            sub_dir = "won"
            if file_type == "actions":
                filename = "actions.txt"
            elif file_type == "response":
                filename = "response.txt"
            else:
                raise ValueError("Invalid file_type for 'won' status. Must be 'actions' or 'response'.")
            sub_dir_path = level_dir / sub_dir
        elif status == "lost":
            if self.refine_enabled and step > 0:
                sub_dir = f"lost_refinement_{step}"
            else:
                sub_dir = "lost"
            if file_type == "prompt":
                filename = "prompt.txt"
            elif file_type == "actions":
                filename = "actions.txt"
            elif file_type == "response":
                filename = "response.txt"
            else:
                raise ValueError("Invalid file_type. Must be 'prompt', 'actions', or 'response'.")
            sub_dir_path = level_dir / sub_dir
        else:
            raise ValueError("Invalid status. Must be 'initial', 'won', or 'lost'.")

        sub_dir_path.mkdir(parents=True, exist_ok=True)
        file_path = sub_dir_path / filename

        with open(file_path, 'w') as f:
            f.write(content)
        print(f"Saved {file_type} to {file_path}")

    def reset(self):
        """Reset the engine and clear the replay buffer."""
        if self.engine:
            self.engine.reset()
            self.replay_buffers = []
            self.actions = []
            print("Engine reset and replay buffer cleared.")
            # Capture and print the initial state
            initial_obs = self.engine.get_obs().copy()
            state = self.format_state(initial_obs)
            self.initial_state = deepcopy(state)
            print(f"Initial state: {self.initial_state}")
        else:
            print("Engine not set. Cannot reset.")

    def initial_request_prompt(self, state, level_id):
        """Generate and save the initial request prompt."""
        domain_content = self.load_file(self.domain_file_name)
        world_model_content = self.load_file(self.world_model_load_name)
        plan = self.get_plan_for_level(level_id)

        prompt = self._make_langchain_prompt(
            text=INITIAL_REQUEST_PROMPT,  # Defined externally
            actions_set=self.actions_set,
            state_format=self.engine.state_format,
            initial_state=state,
            plan=plan,
            domain_file=domain_content,
            world_model=world_model_content,
            utils="directions = {\n    'left': [-1, 0],\n    'right': [1, 0],\n    'up': [0, 1],\n    'down': [0, -1],\n}"
        )

        # Save the initial prompt
        self.save_file(level_id, status="initial", step=0, file_type="prompt", content=prompt.to_string())
        print(f"Initial prompt saved for level {level_id}.")

        # Store the initial state for future refinements
        if self.initial_state is None:
            self.initial_state = deepcopy(state)
            print(f"Initial state set for level {level_id}: {self.initial_state}")

        return prompt

    def extract_explanation(self, response_content):
        """
        Extract explanation from response.txt.
        Explanation is the part that is not within the Python markup tags.
        """
        try:
            explanation = re.sub(r'```Python[\s\S]*?```', '', response_content, flags=re.IGNORECASE).strip()
            if explanation:
                return explanation
            else:
                raise ValueError("No explanation found in the response.")
        except Exception as e:
            print(f"Error extracting explanation: {e}")
            return "No explanation provided."

    def load_all_previous_actions(self, level_id, current_step):
        """
        Load all previous action sequences up to the current refinement step.
        """
        all_actions = []
        for step in range(0, current_step):
            if step == 0:
                # Initial actions
                actions_file = self.save_dir / f"level_{level_id}" / "initial" / "actions.txt"
            else:
                # Refinement steps
                actions_file = self.save_dir / f"level_{level_id}" / f"lost_refinement_{step}" / "actions.txt"
            
            if actions_file.exists():
                with open(actions_file, 'r') as f:
                    try:
                        actions = ast.literal_eval(f.read())
                        all_actions.append(actions)
                    except (SyntaxError, ValueError) as e:
                        print(f"Error loading actions from {actions_file}: {e}")
                        all_actions.append([])
            else:
                print(f"Actions file {actions_file} does not exist. Skipping.")
                all_actions.append([])
        
        return all_actions

    def execute_actions_and_capture_replay(self, actions, level_id):
        """
        Execute a list of actions and capture the replay buffer.
        """
        temp_replay_buffer = []
        for action in actions:
            outcome = self.step_env(action)
            # Capture the replay buffer after each action
            if self.replay_buffers:
                previous_state, executed_action, next_state = self.replay_buffers[-1]
                summary = self._make_observation_summary(previous_state, executed_action, next_state)
                temp_replay_buffer.append({
                    "action": executed_action,
                    "summary": summary
                })
            if outcome == "won":
                print(f"Agent won during execution of actions: {actions}")
                break
            elif outcome == "lost":
                print(f"Agent lost during execution of actions: {actions}")
                break
        return temp_replay_buffer
    
    def generate_action_replay_buffer_pairs(self, level_id, current_step):
        action_replay_pairs = []

        for step in range(1, current_step + 1):  # Include current_step
            if step == 1:
                # Load actions from 'initial' directory for first refinement
                actions_file_path = self.save_dir / f"level_{level_id}" / "initial" / "actions.txt"
            else:
                # Load actions from refinement directories for subsequent refinements
                actions_file_path = self.save_dir / f"level_{level_id}" / f"lost_refinement_{step-1}" / "actions.txt"  # Load the previous step's actions

            # Debugging print to confirm file path
            print(f"Looking for actions file at: {actions_file_path}")

            if not actions_file_path.exists():
                print(f"Actions file not found at {actions_file_path}. Skipping.")
                continue

            # Load actions if the file exists
            try:
                with open(actions_file_path, 'r') as f:
                    actions = ast.literal_eval(f.read())  # Safely read the list of actions
                print(f"Loaded actions for refinement step {step}: {actions}")
            except Exception as e:
                print(f"Error loading actions from {actions_file_path}: {e}")
                actions = []

            # Reset the game engine and replay actions to generate replay buffer
            self.engine.reset()
            self.replay_buffers = []  # Clear the buffer
            for action in actions:
                outcome = self.step_env(action)
                print(f"Executed action '{action}', outcome: {outcome}")

                if outcome == "won":
                    print("Agent won! Exiting.")
                    return True

            replay_buffer_summary = self._make_observation_summaries(self.replay_buffers)
            action_replay_pairs.append((actions, replay_buffer_summary))

        return action_replay_pairs




    
    def format_actions_and_replay_buffers(self, action_replay_pairs):
        """
        Format the action-replay buffer pairs as a string for the prompt.
        """
        formatted_pairs = []

        for idx, (actions, replay_buffer) in enumerate(action_replay_pairs, 1):
            formatted_pair = f"**Previous Actions {idx} and its replay buffer:**\n\nActions:\n{actions}\n\nReplay Buffer:\n"
            formatted_pair += f"{replay_buffer}\n"
            formatted_pairs.append(formatted_pair)

        return "\n".join(formatted_pairs)


    def refine_prompt(self, level_id, refinement_step):
        # Generate action-replay buffer pairs
        action_replay_pairs = self.generate_action_replay_buffer_pairs(level_id, refinement_step)

        # If agent won during replay generation, exit early
        if action_replay_pairs is True:
            print(f"Agent won during replay generation. Exiting refinement process for level {level_id}.")
            return None
        

        
        # Format the action-replay buffer pairs for the prompt
        formatted_action_replay_pairs = self.format_actions_and_replay_buffers(action_replay_pairs)

        history_section = formatted_action_replay_pairs if formatted_action_replay_pairs else "No previous history."


        # Generate the refinement prompt
        prompt = self._make_langchain_prompt(
            text=REFINE_PROMPT,
            actions_set=self.actions_set,
            state_format=self.engine.state_format,
            initial_state=self.initial_state,
            action_replay_tuples=history_section,  # Insert formatted pairs
            utils=self.utils
        )

        # Save the refinement prompt
        self.save_file(level_id, status="lost", step=refinement_step, file_type="prompt", content=prompt.to_string())
        print(f"Refinement prompt saved for refinement step {refinement_step} of level {level_id}.")
        # breakpoint()
        return prompt

    def step_env(self, action):
        """Execute an action in the environment and update the replay buffer with detailed logging."""
        # Capture previous state
        if self.replay_buffers:
            previous_state = self.replay_buffers[-1][2]  # Last next_state
        else:
            previous_state = self.initial_state

        # Execute the action
        self.engine.step(action)

        # Capture next state
        state_after_action = deepcopy(self.engine.get_obs())
        state_after_action = self.format_state(state_after_action)

        # Append the transition to replay buffer
        self.replay_buffers.append((deepcopy(previous_state), action, deepcopy(state_after_action)))

        # Update actions list
        self.actions.append(action)

        # Check for win/loss
        print(f"After action '{action}', state: {state_after_action}")  # Debug statement
        if self.engine.won:
            print("Agent won!")
            return "won"
        elif self.engine.lost:
            print("Agent lost.")
            return "lost"
        return None  # Continue

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
                            if (x not in val1):
                                removed.append(x)
                    string += f'"{key}": Added: {added}\n'
                    string += f'"{key}": Removed: {removed}\n'
                else:
                    string += f'"{key}": {_stringify(val0)} --> {_stringify(val1)}\n'

        return string

    def _make_observation_summary(self, state0, action, state1):
        """
        Create a single observation summary without prediction errors.
        """
        summary_changes = self._get_state_deltas_str(state0, state1)
        return (
            f"Initial state: {state0}\n"
            f"Action: {action}\n"
            f"Next state: {state1}\n"
            f"Summary of changes:\n{summary_changes}"
        )

    def _make_observation_summaries(self, replay_buffers):
        """
        Create a summary of the replay buffer transitions.
        """
        summaries = []
        for obs in replay_buffers:
            state0, action, state1 = obs
            summary = self._make_observation_summary(state0, action, state1)
            summaries.append(summary)
        return "\n\n".join(summaries)

    def _eq(self, x, y):
        """
        Recursively convert lists to tuples and compare for equality.
        """
        def deep_convert_to_tuple(v):
            if isinstance(v, list):
                return tuple(deep_convert_to_tuple(i) for i in v)
            return v

        x_converted = deep_convert_to_tuple(x)
        y_converted = deep_convert_to_tuple(y)
        return x_converted == y_converted

    def find_latest_refinement(self, level_id):
        """Find the latest refinement step in the folder."""
        level_dir = self.save_dir / f"level_{level_id}"
        refinements = [d for d in level_dir.iterdir() if d.is_dir() and 'lost_refinement' in d.name]
        
        if refinements:
            # Extract the refinement step numbers from the folder names
            refinement_steps = [int(d.name.split('_')[-1]) for d in refinements if d.name.split('_')[-1].isdigit()]
            if refinement_steps:
                return max(refinement_steps)  # Return the highest refinement step
        return 0  # No refinements exist yet


    def run(self, engine, level_id):
        """Run the initial request to get action sequence from LLM and evaluate it."""
        self.engine = engine  # Assign engine to self

        # Reset the engine and get the initial state
        self.reset()

        # If refinement is disabled, perform only the initial guess
        if not self.refine_enabled:
            prompt = self.initial_request_prompt(self.initial_state, level_id)
            print(f"Sending initial request for level {level_id}...")

            if self.groq:
                print(f"Running with Groq for level {level_id}...")
                prompt_content = f"""
                You are an AI agent that must come up with a list of actions that need to be taken to win a certain level in a game. 
                These actions can only come from the action space given below. You are given an example of what your response 
                format for this list of actions should look like. You will also need to provide your reasoning for why this will allow you to win.

                You are given your current state that you start from in the level. 

                So using the information please return the action sequence that will result in winning the level. 
                Make sure to give your explanation, also
                make sure to just have a sepearte section with your actions as demonstrated in the response format.

                ACTION SPACE (YOUR LIST SHOULD BE COMPOSED OF THESE ACTIONS):

                {self.actions_set}

                STATE FORMAT:

                {self.engine.state_format}

                INITIAL STATE:

                {self.initial_state}

                UTILS:

                {self.utils}

                RESPONSE FORMAT (just a random example list, make sure your answer is returned with markup tag):

                ```Python

                ["right", "left", "up", "down"]

                ```

                explanation:

                Example explanation.
                """
                response = self.groq_query(prompt_content)
            else:
                response = self.query_lm(prompt)
            
            print(f"Received response for initial guess: {response}")

            # Save the initial response
            self.save_file(level_id, status="initial", step=0, file_type="response", content=response)

            # Extract action set from response
            actions = self.extract_actions(response)
            print(f"Extracted actions for initial guess: {actions}")

            # Save the initial actions in compact format
            actions_str = str(actions)
            self.save_file(level_id, status="initial", step=0, file_type="actions", content=actions_str)

            # Execute initial actions
            for action in actions:
                print(f"Executing action: {action}")
                outcome = self.step_env(action)

                if outcome == "won":
                    print(f"Level {level_id} won with initial actions.")
                    # Actions and response already saved
                    return True
                elif outcome == "lost":
                    print(f"Level {level_id} lost with initial actions.")
                    break  # Proceed to refinement if enabled

            # After executing all actions, check if the game is completed without win/loss
            if not self.engine.won and not self.engine.lost:
                print(f"Level {level_id} completed without win/loss.")
                # Treat 'completed' as 'lost'
                self.save_file(level_id, status="lost", step=0, file_type="response", content=response)
                self.save_file(level_id, status="lost", step=0, file_type="actions", content=actions_str)

            return False  # End if no refinement

        else:
            # Refinement enabled, pick up from the latest refinement step
            latest_refinement = self.find_latest_refinement(level_id)
            print(f"Resuming from refinement step {latest_refinement + 1}...")

            for refinement_step in range(latest_refinement + 1, self.max_refinements + 1):
                print(f"Starting refinement step {refinement_step} for level {level_id}...")

                # Generate the refinement prompt with all previous actions and replay buffers
                prompt = self.refine_prompt(level_id, refinement_step)

                # If prompt is None, it means the agent has already won, exit the loop
                if prompt is None:
                    print(f"Refinement process exited early for level {level_id} after winning.")
                    return True

                print(f"Sending refinement request {refinement_step} for level {level_id}...")

                
                if self.groq:
                    print(f"Running with Groq for level {level_id}...")
                    prompt_content = f"""
                    Prompt content with relevant parameters filled:

                    ACTION SPACE:
                    {self.actions_set}

                    INITIAL STATE:
                    {self.initial_state}

                    UTILS:
                    {self.utils}
                    """
                    response = self.groq_query(prompt_content)
                else:
                    print(f"Sending initial request for level {level_id}...")
                    response = self.query_lm(prompt)

                print(f"Received response: {response}")


                # Query the LLM for refined actions
                response = self.query_lm(prompt)
                print(f"Received response for refinement {refinement_step}: {response}")

                # Save the refinement response
                self.save_file(level_id, status="lost", step=refinement_step, file_type="response", content=response)

                # Extract refined actions
                actions = self.extract_actions(response)
                print(f"Extracted actions for refinement {refinement_step}: {actions}")

                # Save the refined actions in compact format
                actions_str = str(actions)
                self.save_file(level_id, status="lost", step=refinement_step, file_type="actions", content=actions_str)

                # Execute refined actions
                for action in actions:
                    print(f"Executing refined action: {action}")
                    outcome = self.step_env(action)

                    if outcome == "won":
                        print(f"Level {level_id} won at refinement step {refinement_step}.")
                        return True
                    elif outcome == "lost":
                        print(f"Level {level_id} lost at refinement step {refinement_step}.")
                        break  # Proceed to next refinement step if any

                # After executing all refined actions, check if the game is completed without win/loss
                if not self.engine.won and not self.engine.lost:
                    print(f"Level {level_id} completed without win/loss at refinement step {refinement_step}.")
                    # Treat 'completed' as 'lost'
                    self.save_file(level_id, status="lost", step=refinement_step, file_type="response", content=response)
                    self.save_file(level_id, status="lost", step=refinement_step, file_type="actions", content=actions_str)

            print(f"Max refinements reached for level {level_id}.")
            return False



    def extract_actions(self, response):
        """Extract the action list from the LLM response."""
        try:
            # Use regular expression to find Python code block
            code_block_match = re.findall(r'```Python([\s\S]*?)```', response, re.IGNORECASE)
            if code_block_match:
                # Extract the actions from the code block
                code_block = code_block_match[0].strip()
                actions = ast.literal_eval(code_block)
                # Ensure only valid actions (right, left, up, down) are returned
                actions = [action for action in actions if action in self.actions_set]
                return actions
            else:
                raise ValueError("No properly formatted Python code block found.")
        except (SyntaxError, ValueError, IndexError) as e:
            print(f"Error extracting actions: {e}")
            return []  # Return empty list if extraction fails

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, default='baba')
    parser.add_argument('--levels', type=str, default="[('demo_LEVELS', 0)]")  # Example format
    parser.add_argument('--episode-length', type=int, default=20)
    parser.add_argument('--world-model-file-name', type=str, default='worldmodel.py')
    parser.add_argument('--domain-file-name', type=str, default='domain.pddl')
    parser.add_argument('--predicates-file-name', type=str, default='predicates.py')
    parser.add_argument('--plans-json-path', type=str, default='plans.json', help="Path to the JSON file containing plans for each level.")
    parser.add_argument('--refine', action='store_true', help="Enable refinement if the LLM's action sequence leads to a loss")
    parser.add_argument('--max-refinements', type=int, default=5, help="Maximum number of refinement steps allowed")
    parser.add_argument('--save-dir', type=str, default='groq_baselines', help="Directory to save the results")
    parser.add_argument('--groq', action='store_true', help="Use Groq-specific querying instead of standard LM querying")

    args = parser.parse_args()
    levels = eval(args.levels)

    for level_set, level_id in levels:
        if args.game == 'baba':
            engine = BabaIsYou(level_set=level_set, level_id=level_id)

        agent = Baselines(
            episode_length=args.episode_length,
            world_model_load_name=args.world_model_file_name,
            domain_file_name=args.domain_file_name,
            predicates_file_name=args.predicates_file_name,
            refine=args.refine,
            max_refinements=args.max_refinements,
            save_dir=args.save_dir,
            plans_json_path=args.plans_json_path,
            groq=args.groq
        )
        agent.run(engine, level_id)