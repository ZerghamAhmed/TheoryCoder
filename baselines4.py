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

You previously attemped this level and returned the following action sequence but did not win the game. 
Your state transition information from the replay buffer is given below and may help you understand how to provide a corrected action sequence.
You are also given previous things that you noticed about the domain that may help you retrun the correct actions.

Please provide your corrected action sequence that will result in winning the level. 
Also for your explanation, you should mention why your previous predicted action sequence did not win the game.

ACTION SPACE (YOUR LIST SHOULD BE COMPOSED OF THESE ACTIONS):

{actions_set}

STATE FORMAT:

{state_format}

INITIAL STATE FOR LEVEL:

{initial_state}

PREVIOUS ACTION SEQUENCE PREDICTION:

{previous_guessed_actions}

REPLAY BUFFER:

{replay_buffer_summary}

PREVIOUS THINGS YOU NOTICED ABOUT THE DOMAIN:

{explanation}


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

class Baselines:
    def __init__(self, episode_length, world_model_load_name, domain_file_name, predicates_file_name,
                 refine=False, max_refinements=5, save_dir="experiment_results", plans_json_path="plans.json"):
        self.episode_length = episode_length
        self.world_model_load_name = world_model_load_name
        self.domain_file_name = domain_file_name
        self.predicates_file_name = predicates_file_name
        self.refine_enabled = refine
        self.max_refinements = max_refinements
        self.save_dir = Path(save_dir).resolve()
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.language_model = 'o1-preview'  # Update as needed
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
            state_format=engine.state_format,
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
            explanation = re.sub(r'```Python[\s\S]*?```', '', response_content).strip()
            if explanation:
                return explanation
            else:
                raise ValueError("No explanation found in the response.")
        except Exception as e:
            print(f"Error extracting explanation: {e}")
            return ""

    def load_previous_prompt_and_replay(self, level_id, step):
        """
        Load the previous prompt.txt and extract the replay buffer from it.
        """
        try:
            prompt_file_path = self.save_dir / f"level_{level_id}" / f"lost_refinement_{step}" / "prompt.txt"
            with open(prompt_file_path, 'r') as f:
                prompt_content = f.read()

            # Extract the previous replay buffer from the prompt file
            replay_buffer_match = re.search(r'REPLAY BUFFER:\n([\s\S]*)UTILS:', prompt_content)
            if replay_buffer_match:
                replay_buffer = replay_buffer_match.group(1).strip()
                return replay_buffer
            else:
                raise ValueError("No replay buffer found in the prompt.")
        except FileNotFoundError:
            print(f"Previous prompt file {prompt_file_path} not found.")
            return ""
        except Exception as e:
            print(f"Error loading previous replay buffer: {e}")
            return ""

    def accumulate_explanations(self, level_id, current_step):
        """
        Accumulate explanations from all previous refinements.
        If a refinement file is missing, it will skip that file.
        """
        explanations = []
        for step in range(1, current_step):
            try:
                # Load the response file for the current refinement step
                response_file_path = self.save_dir / f"level_{level_id}" / f"lost_refinement_{step}" / "response.txt"
                if response_file_path.exists():
                    response_content = self.load_file(response_file_path)
                    explanation = self.extract_explanation(response_content)
                    explanations.append(explanation)
                else:
                    print(f"Response file for refinement step {step} not found. Skipping.")
            except Exception as e:
                print(f"Error loading explanation from step {step}: {e}")
                continue  # If any issue occurs, continue to the next step
        
        # Join all accumulated explanations
        accumulated_explanation = "\n".join(explanations).strip()
        return accumulated_explanation if accumulated_explanation else "No previous explanations found."

    def accumulate_replay_buffers(self, level_id, current_step):
        """
        Accumulate replay buffers from all previous refinements.
        If a prompt file is missing, it will skip that file.
        """
        replay_buffers = []
        for step in range(1, current_step):
            try:
                prompt_file_path = self.save_dir / f"level_{level_id}" / f"lost_refinement_{step}" / "prompt.txt"
                if prompt_file_path.exists():
                    prompt_content = self.load_file(prompt_file_path)
                    replay_buffer_match = re.search(r'REPLAY BUFFER:\n([\s\S]*)UTILS:', prompt_content)
                    if replay_buffer_match:
                        replay_buffer = replay_buffer_match.group(1).strip()
                        replay_buffers.append(replay_buffer)
                    else:
                        print(f"No replay buffer found in prompt for step {step}.")
                else:
                    print(f"Prompt file for refinement step {step} not found. Skipping.")
            except Exception as e:
                print(f"Error loading replay buffer from step {step}: {e}")
                continue  # If any issue occurs, continue to the next step

        # Join all accumulated replay buffers
        accumulated_replay_buffer = "\n\n".join(replay_buffers).strip()
        return accumulated_replay_buffer if accumulated_replay_buffer else "No previous replay buffers found."

    def refine_prompt(self, previous_actions, replay_buffer_summary, level_id, refinement_step):
        """Generate and save the refinement request prompt, accumulating explanations and replay buffers."""
        
        # Accumulate all previous explanations
        accumulated_explanation = self.accumulate_explanations(level_id, refinement_step)
        
        # Accumulate all previous replay buffers
        accumulated_replay_buffer = self.accumulate_replay_buffers(level_id, refinement_step)

        # Update the refinement prompt to include the accumulated explanation and replay buffer
        prompt = self._make_langchain_prompt(
            text=REFINE_PROMPT,  # Defined externally
            actions_set=self.actions_set,
            state_format=self.engine.state_format,
            initial_state=self.initial_state,
            previous_guessed_actions=previous_actions,
            replay_buffer_summary=replay_buffer_summary,
            explanation=accumulated_explanation,
            previous_replay_buffer=accumulated_replay_buffer,
            utils=self.utils
        )

        # Save the refinement prompt
        self.save_file(level_id, status="lost", step=refinement_step, file_type="prompt", content=prompt.to_string())
        print(f"Refinement prompt saved for refinement step {refinement_step} of level {level_id}.")
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
                            if x not in val1:
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

    def refine(self, level_id):
        """Handle refinement steps."""
        for refinement_step in range(1, self.max_refinements + 1):
            print(f"Starting refinement step {refinement_step} for level {level_id}...")

            self.reset()

            # Determine the previous actions file
            if refinement_step == 1:
                previous_actions_file = self.save_dir / f"level_{level_id}" / "initial" / "actions.txt"
            else:
                previous_actions_file = self.save_dir / f"level_{level_id}" / f"lost_refinement_{refinement_step - 1}" / "actions.txt"

            if not previous_actions_file.exists():
                print(f"Previous actions file {previous_actions_file} does not exist. Stopping refinements.")
                break

            with open(previous_actions_file, 'r') as f:
                try:
                    previous_actions = ast.literal_eval(f.read())
                except (SyntaxError, ValueError) as e:
                    print(f"Error loading previous actions from {previous_actions_file}: {e}")
                    break

            # Execute previous actions to populate replay buffer
            print(f"Re-executing previous actions from {previous_actions_file} to update replay buffer...")
            for action in previous_actions:
                outcome = self.step_env(action)
                if outcome in ["won", "lost"]:
                    break  # Stop executing if won during replay w/ previous actions

            # Generate the replay buffer summary
            replay_buffer_summary = self._make_observation_summaries(self.replay_buffers)
            print(f"Replay buffer summary for refinement step {refinement_step}:\n{replay_buffer_summary}")

            # Generate the refinement prompt
            prompt = self.refine_prompt(
                previous_actions=previous_actions,
                replay_buffer_summary=replay_buffer_summary,
                level_id=level_id,
                refinement_step=refinement_step
            )

            # breakpoint()

            # Clear the replay buffer after generating the prompt
            self.replay_buffers = []
            self.actions = []
            print("Replay buffer cleared for next refinement.")

            print(f"Sending refinement request {refinement_step} for level {level_id}...")

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

    def run(self, engine, level_id):
        """Run the initial request to get action sequence from LLM and evaluate it."""
        self.engine = engine  # Assign engine to self

        # Reset the engine and get the initial state
        self.reset()

        # If refinement is disabled, perform initial guess only
        if not self.refine_enabled:
            # Perform Initial Guess
            prompt = self.initial_request_prompt(self.initial_state, level_id)
            print(f"Sending initial request for level {level_id}...")

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

            # Proceed to Refinement if enabled
            if self.refine_enabled and (self.engine.lost or not self.engine.won):
                refinement_success = self.refine(level_id)
                if refinement_success:
                    return True
            return False
        else:
            # Perform Refinements only
            refinement_success = self.refine(level_id)
            return refinement_success

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

# Example usage (assuming prompts are defined externally):
# INITIAL_REQUEST_PROMPT = "Your initial prompt here..."
# REFINE_PROMPT = "Your refinement prompt here..."

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, default='baba')
    parser.add_argument('--levels', type=str, default="[('demo_LEVELS', 13)]")  # Example format
    parser.add_argument('--episode-length', type=int, default=20)
    parser.add_argument('--world-model-file-name', type=str, default='worldmodel.py')
    parser.add_argument('--domain-file-name', type=str, default='domain.pddl')
    parser.add_argument('--predicates-file-name', type=str, default='predicates.py')
    parser.add_argument('--plans-json-path', type=str, default='plans.json', help="Path to the JSON file containing plans for each level.")
    parser.add_argument('--refine', action='store_true', help="Enable refinement if the LLM's action sequence leads to a loss")
    parser.add_argument('--max-refinements', type=int, default=5, help="Maximum number of refinement steps allowed")
    parser.add_argument('--save-dir', type=str, default='o1_preview_4', help="Directory to save the results")

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
            plans_json_path=args.plans_json_path
        )
        agent.run(engine, level_id)
