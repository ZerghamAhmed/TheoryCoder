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
You are an AI agent that must come up with actions that need to be taken to win the game. 
You are given a model of the game in the form of a python program that captures the logic and mechanics of the game. 
You are given the possible actions you can take, the format of the state, the current state, 
a high-level PDDL domain file which gives you information about the high-level plans and mechanics of the domain, 
and a low-level python transition function which gives you detailed information about the mechanics of the domain. 
Please return the action sequence that will result in winning the level. If you have an explanation give it separately.
Make sure to just have a sepearte section with your actions as demonstrated in the response format.

ACTION SPACE:

{actions_set}

STATE FORMAT:

{state_format}

CURRENT STATE:

{current_state}

HIGH-LEVEL DOMAIN FILE:

{domain_file}

CURRENT LOW-LEVEL WORLD MODEL:

{world_model}

UTILS:

{utils}

RESPONSE FORMAT (just a random example list, make sure your answer is returned with markup tag):

```Python

["right", "left", "up", "down"]

```

"""

REFINE_PROMPT = """
You are an AI agent that must come up with actions that need to be taken to win the game. 
You are given a model of the game in the form of a python program that captures the logic and mechanics of the game. 
You are given the possible actions you can take, the format of the state, the current state, 
a high-level PDDL domain file which gives you information about the high-level plans and mechanics of the domain, 
and a low-level python transition function which gives you detailed information about the mechanics of the domain. 
Please return the action sequence that will result in winning the level. 
You previously returned the following action sequence but did not win the game. 
Your state transition information is given below and may help you understand how to provide a corrected action sequence.

ACTION SPACE:

{actions_set}

STATE FORMAT:

{state_format}

INITIAL STATE FOR LEVEL:

{starting_state}

PREVIOUS ACTION SEQUENCE PREDICTION:

{previous_guessed_actions}

REPLAY BUFFER:

{replay_buffer_summary}

HIGH-LEVEL DOMAIN FILE:

{domain_file}

CURRENT LOW-LEVEL WORLD MODEL:

{world_model}

UTILS:

{utils}

RESPONSE FORMAT (just a random example list, make sure your answer is returned with markup tag, explanations should be outside it):

```Python

["right", "left", "up", "down"]

```

explanation:

Example explanation.

"""

import ast
import json
import re
from pathlib import Path
from copy import deepcopy
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI
from games import BabaIsYou
from predicates import rule_formed


class Baselines:
    def __init__(self, episode_length, world_model_load_name, domain_file_name, predicates_file_name,
                 refine=False, max_refinements=2, save_dir="experiment_results"):
        self.episode_length = episode_length
        self.world_model_load_name = world_model_load_name
        self.domain_file_name = domain_file_name
        self.predicates_file_name = predicates_file_name
        self.refine_enabled = refine  # Control refinement
        self.max_refinements = max_refinements  # Maximum number of refinements allowed
        self.save_dir = Path(save_dir).resolve()  # Absolute path for saving results
        self.save_dir.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist

        self.language_model = 'gpt-4o-mini'  # Update as needed
        self.chat = ChatOpenAI(model_name=self.language_model, temperature=1.0)  # Correct model initialization
        self.query_lm = lambda prompt: self.chat.invoke(prompt.to_messages()).content  # Use invoke as per deprecation

        self.tape = []  # Store all interactions
        self.replay_buffers = []  # Replay buffer to store state transitions
        self.actions_set = ["up", "down", "left", "right"]
        self.utils = {
            'directions': {
                'left': [-1, 0],
                'right': [1, 0],
                'up': [0, 1],
                'down': [0, -1],
            }
        }
        self.actions = []  # Track executed actions
        self.initial_state = None  # Store the initial state of the level

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

        # Add controllables based on "is you" rule
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
        level_dir.mkdir(parents=True, exist_ok=True)  # Ensure level directory exists

        if status == "initial":
            # Initial guess
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
            # Win outcome
            sub_dir = "won"
            if file_type == "actions":
                filename = "actions.txt"
            elif file_type == "response":
                filename = "response.txt"
            else:
                raise ValueError("Invalid file_type for 'won' status. Must be 'actions' or 'response'.")
            sub_dir_path = level_dir / sub_dir
        elif status == "lost":
            # Refinement steps or initial loss without refinement
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

        sub_dir_path.mkdir(parents=True, exist_ok=True)  # Ensure subdirectory exists
        file_path = sub_dir_path / filename

        with open(file_path, 'w') as f:
            f.write(content)
        print(f"Saved {file_type} to {file_path}")

    def initial_request_prompt(self, state, level_id):
        """Generate and save the initial request prompt."""
        domain_content = self.load_file(self.domain_file_name)
        world_model_content = self.load_file(self.world_model_load_name)

        prompt = self._make_langchain_prompt(
            text=INITIAL_REQUEST_PROMPT,  # Reference to external prompt
            actions_set=self.actions_set,
            state_format=engine.state_format,
            current_state=state,
            domain_file=domain_content,
            world_model=world_model_content,
            utils="directions = {\n    'left': [-1, 0],\n    'right': [1, 0],\n    'up': [0, 1],\n    'down': [0, -1],\n}"
        )

        # Save the initial prompt
        self.save_file(level_id, status="initial", step=0, file_type="prompt", content=prompt.to_string())

        # Store the initial state for future refinements
        if self.initial_state is None:
            self.initial_state = deepcopy(state)

        return prompt

    def refine_prompt(self, previous_actions, replay_buffer_summary, level_id, refinement_step):
        """Generate and save the refinement request prompt."""
        domain_content = self.load_file(self.domain_file_name)
        world_model_content = self.load_file(self.world_model_load_name)

        prompt = self._make_langchain_prompt(
            text=REFINE_PROMPT,  # Reference to external prompt
            actions_set=self.actions_set,
            state_format=engine.state_format,
            starting_state=self.initial_state,  # Use the stored initial state
            previous_guessed_actions=previous_actions,
            replay_buffer_summary=replay_buffer_summary,  # Ensure correct keyword
            domain_file=domain_content,
            world_model=world_model_content,
            utils="directions = {\n    'left': [-1, 0],\n    'right': [1, 0],\n    'up': [0, 1],\n    'down': [0, -1],\n}"
        )

        # Save the refinement prompt
        self.save_file(level_id, status="lost", step=refinement_step, file_type="prompt", content=prompt.to_string())

        return prompt

    def step_env(self, engine, action):
        """Step the game engine, store the transition in the replay buffer."""
        # Step the game engine with the given action
        engine.step(action)

        # Deep copy the current state after the action
        state_after_action = deepcopy(engine.get_obs())
        state_after_action = {key: [list(item) for item in value] if isinstance(value, list) else value for key, value in state_after_action.items()}

        # Identify controllables based on "is you" rule
        controllables = {
            entity for entity in state_after_action
            if rule_formed(state_after_action, f'{entity[:-4]}_word', 'is_word', 'you_word')
        }

        # Add controllables to the state dictionary
        state_after_action['controllables'] = list(controllables)

        # Append the action to the actions list
        self.actions.append(action)

        # Determine the previous state
        if self.replay_buffers:
            previous_state = self.replay_buffers[-1][2]  # Last next_state
        else:
            previous_state = self.initial_state  # Use the stored initial state

        # Append the transition to replay_buffers
        self.replay_buffers.append((previous_state, action, state_after_action))

        # Check for win/loss
        if engine.won:
            print("Agent won!")
            return "won"
        elif engine.lost:
            print("Agent lost.")
            return "lost"
        return None  # Continue the loop if neither win nor loss

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

    def run(self, engine, level_id):
        """Run the initial request to get action sequence from LLM and evaluate it."""
        self.engine = engine  # Assign engine to self

        # Reset the engine and get the initial state
        engine.reset()
        initial_obs = engine.get_obs().copy()
        state = self.format_state(initial_obs)
        self.initial_state = deepcopy(state)

        if not self.refine_enabled:
            # Perform Initial Guess
            prompt = self.initial_request_prompt(state, level_id)
            print(f"Sending initial request for level {level_id}...")

            response = self.query_lm(prompt)
            print(f"Received response for initial guess: {response}")

            # Save the initial response
            self.save_file(level_id, status="initial", step=0, file_type="response", content=response)

            # Extract action set from response
            actions = self.extract_actions(response)
            print(f"Extracted actions for initial guess: {actions}")

            # Save the initial actions in compact format
            actions_str = str(actions)  # Convert list to compact string
            self.save_file(level_id, status="initial", step=0, file_type="actions", content=actions_str)

            # Execute initial actions
            for action in actions:
                print(f"Executing action: {action}")
                outcome = self.step_env(engine, action)

                if outcome == "won":
                    print(f"Level {level_id} won with initial actions.")
                    # Actions and response already saved
                    return True
                elif outcome == "lost":
                    print(f"Level {level_id} lost with initial actions.")
                    break  # Proceed to refinement if enabled

            # After executing all actions, check if the game is completed without win/loss
            if not engine.won and not engine.lost:
                print(f"Level {level_id} completed without win/loss.")
                # Treat 'completed' as 'lost'
                self.save_file(level_id, status="lost", step=0, file_type="response", content=response)
                self.save_file(level_id, status="lost", step=0, file_type="actions", content=actions_str)

            # Proceed to Refinement if enabled
            if self.refine_enabled and (engine.lost or not engine.won):
                self.refine(engine, level_id)
            return False
        else:
            # Perform Refinements only
            self.refine(engine, level_id)
            return False

    def refine(self, engine, level_id):
        """Refine actions if agent loses or fails to win the game."""
        for refinement_step in range(1, self.max_refinements + 1):
            print(f"Starting refinement step {refinement_step} for level {level_id}...")

            # Reset the engine to the initial state
            engine.reset()
            initial_obs = engine.get_obs().copy()
            state = self.format_state(initial_obs)
            self.initial_state = deepcopy(state)

            # Determine the previous actions file
            if refinement_step == 1:
                previous_actions_file = self.save_dir / f"level_{level_id}" / "initial/actions.txt"
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

            # Execute previous actions to update the replay buffer
            print(f"Re-executing previous actions from {previous_actions_file} to update replay buffer...")
            for action in previous_actions:
                outcome = self.step_env(engine, action)
                if outcome in ["won", "lost"]:
                    break  # Stop executing if won or lost during replay

            # Generate the replay buffer summary
            replay_buffer_summary = self._make_observation_summaries(self.replay_buffers)

            # Generate the refinement prompt
            prompt = self.refine_prompt(
                previous_actions=previous_actions,
                replay_buffer_summary=replay_buffer_summary,  # Corrected keyword
                level_id=level_id,
                refinement_step=refinement_step
            )

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
            actions_str = str(actions)  # Convert list to compact string
            self.save_file(level_id, status="lost", step=refinement_step, file_type="actions", content=actions_str)

            # Execute refined actions
            for action in actions:
                print(f"Executing refined action: {action}")
                outcome = self.step_env(engine, action)

                if outcome == "won":
                    print(f"Level {level_id} won at refinement step {refinement_step}.")
                    return True
                elif outcome == "lost":
                    print(f"Level {level_id} lost at refinement step {refinement_step}.")
                    break  # Proceed to next refinement step if any

            # After executing all refined actions, check if the game is completed without win/loss
            if not engine.won and not engine.lost:
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
    parser.add_argument('--levels', type=str, default="[('demo_LEVELS', 0)]")  # Changed to level 0 for consistency
    parser.add_argument('--episode-length', type=int, default=20)
    parser.add_argument('--world-model-file-name', type=str, default='worldmodel.py')
    parser.add_argument('--domain-file-name', type=str, default='domain.pddl')
    parser.add_argument('--predicates-file-name', type=str, default='predicates.py')
    parser.add_argument('--json-reporter-path', type=str, default='KekeCompetition-main/Keke_JS/reports/TBRL_BABA_REPORT.json')
    parser.add_argument('--refine', action='store_true', help="Enable refinement if the LLM's action sequence leads to a loss")
    parser.add_argument('--max-refinements', type=int, default=2, help="Maximum number of refinement steps allowed")
    parser.add_argument('--save-dir', type=str, default='experiment_results', help="Directory to save the results")

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
            save_dir=args.save_dir  # Pass the save directory
        )
        agent.run(engine, level_id)
