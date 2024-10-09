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

PREVIOUS ACTION SEQUENCE PREDICTION:

{previous_guessed_actions}

REPLAY BUFFER:

{replay_buffer}

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
            filename = f"prompt.txt" if file_type == "prompt" else \
                       f"actions.txt" if file_type == "actions" else \
                       f"response.txt"
            sub_dir_path = level_dir / sub_dir
        elif status == "won":
            # Win outcome
            sub_dir = "won"
            filename = f"won_actions.txt" if file_type == "actions" else \
                       f"won_response.txt"
            sub_dir_path = level_dir / sub_dir
        elif status == "lost":
            # Refinement steps
            sub_dir = f"lost_refinement_{step}"
            filename = f"prompt.txt" if file_type == "prompt" else \
                       f"actions.txt" if file_type == "actions" else \
                       f"response.txt"
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

        formatted_state = json.dumps(state, indent=4)
        prompt = self._make_langchain_prompt(INITIAL_REQUEST_PROMPT,
            actions_set=self.actions_set,
            state_format=formatted_state,
            current_state=formatted_state,
            domain_file=domain_content,
            world_model=world_model_content,
            utils="directions = {\n    'left': [-1, 0],\n    'right': [1, 0],\n    'up': [0, 1],\n    'down': [0, -1],\n}"
        )

        # Save the initial prompt
        self.save_file(level_id, status="initial", step=0, file_type="prompt", content=prompt.to_string())

        return prompt

    def refine_prompt(self, state, previous_actions, replay_buffer, level_id, refinement_step):
        """Generate and save the refinement request prompt."""
        domain_content = self.load_file(self.domain_file_name)
        world_model_content = self.load_file(self.world_model_load_name)

        formatted_state = json.dumps(state, indent=4)
        formatted_replay = json.dumps(replay_buffer, indent=4)
        formatted_previous_actions = json.dumps(previous_actions, indent=4)

        prompt = self._make_langchain_prompt(REFINE_PROMPT,
            actions_set=self.actions_set,
            state_format=formatted_state,
            previous_guessed_actions=formatted_previous_actions,
            replay_buffer=formatted_replay,
            domain_file=domain_content,
            world_model=world_model_content,
            utils="directions = {\n    'left': [-1, 0],\n    'right': [1, 0],\n    'up': [0, 1],\n    'down': [0, -1],\n}"
        )

        breakpoint()

        # Save the refinement prompt
        self.save_file(level_id, status="lost", step=refinement_step, file_type="prompt", content=prompt.to_string())

        return prompt

    def step_env(self, engine, action):
        """Step the game engine, store the transition in the replay buffer."""
        # Step the game engine with the given action
        engine.step(action)

        # Deep copy observations before and after the action
        state = deepcopy(engine.get_obs())
        state = {key: [list(item) for item in value] if isinstance(value, list) else value for key, value in state.items()}

        # Identify controllables based on "is you" rule
        controllables = {
            entity for entity in state
            if rule_formed(state, f'{entity[:-4]}_word', 'is_word', 'you_word')
        }

        # Add controllables to the state dictionary
        state['controllables'] = list(controllables)

        # Append the action to the actions list
        self.actions.append(action)

        # Update the replay buffer with (state, action, next_state)
        # Assuming `engine.get_obs()` now returns the next state after the action
        # If `state` is the current state before the action, then `next_state` is after
        # Since we have only one `get_obs()`, we'll treat `state` as the current state after action
        # To get the previous state, you might need to store it before stepping
        # For simplicity, let's assume `state_before_action` is accessible
        # Here, since we don't have it, we'll store only the current state
        # Adjust as necessary based on your engine's capabilities

        # For accurate state-action-next_state, you need to track state before action
        # Here, we'll assume that `state_before_action` was already stored in `self.replay_buffers[-1][0]`
        # If `self.replay_buffers` is empty, we cannot append a proper tuple
        if self.replay_buffers:
            state_before_action = self.replay_buffers[-1][2]  # Last next_state
        else:
            # If replay_buffers is empty, assume the initial state
            state_before_action = state

        # Append the transition to replay_buffers
        self.replay_buffers.append((state_before_action, action, state))

        # Check for win/loss
        if engine.won:
            print("Agent won!")
            return "won"
        elif engine.lost:
            print("Agent lost.")
            return "lost"
        return None  # Continue the loop if neither win nor loss

    def run(self, engine, level_id):
        """Run the initial request to get action sequence from LLM and evaluate it."""
        self.engine = engine  # Assign engine to self
        state = self.format_state(deepcopy(engine.get_obs()))  # Format the initial state

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

            # Save the initial actions
            self.save_file(level_id, status="initial", step=0, file_type="actions", content=json.dumps(actions, indent=4))

            # Execute initial actions
            for action in actions:
                print(f"Executing action: {action}")
                outcome = self.step_env(engine, action)

                if outcome == "won":
                    # Save actions and response already done
                    print(f"Level {level_id} won with initial actions.")
                    return True
                elif outcome == "lost":
                    print(f"Level {level_id} lost with initial actions.")
                    # Save the lost outcome
                    # This has already been handled in step_env
                    break
                # If outcome is None, continue executing remaining actions

            # After executing all actions, check if the game is completed without win/loss
            if not engine.won and not engine.lost:
                print(f"Level {level_id} completed without win/loss.")
                # Treat 'completed' as 'lost'
                # Save the 'completed' outcome as 'lost'
                self.save_file(level_id, status="lost", step=0, file_type="response", content=response)
                self.save_file(level_id, status="lost", step=0, file_type="actions", content=json.dumps(actions, indent=4))

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
            # Determine the previous actions file
            if refinement_step == 1:
                previous_actions_file = self.save_dir / f"level_{level_id}" / "initial/actions.txt"
            else:
                previous_actions_file = self.save_dir / f"level_{level_id}" / f"lost_refinement_{refinement_step - 1}" / "actions.txt"

            if not previous_actions_file.exists():
                print(f"Previous actions file {previous_actions_file} does not exist. Stopping refinements.")
                break

            with open(previous_actions_file, 'r') as f:
                previous_actions = json.load(f)

            # Execute previous actions to update the replay buffer
            print(f"Re-executing previous actions from {previous_actions_file} to update replay buffer...")
            for action in previous_actions:
                outcome = self.step_env(engine, action)
                if outcome == "won" or outcome == "lost":
                    break  # Stop executing if won or lost during replay

            # Generate the refinement prompt
            current_state = self.format_state(deepcopy(engine.get_obs()))
            prompt = self.refine_prompt(current_state, previous_actions, self.replay_buffers, level_id, refinement_step)

            print(f"Sending refinement request {refinement_step} for level {level_id}...")

            # Query the LLM for refined actions
            response = self.query_lm(prompt)
            print(f"Received response for refinement {refinement_step}: {response}")

            # Save the refinement response
            self.save_file(level_id, status="lost", step=refinement_step, file_type="response", content=response)

            # Extract refined actions
            actions = self.extract_actions(response)
            print(f"Extracted actions for refinement {refinement_step}: {actions}")

            # Save the refined actions
            self.save_file(level_id, status="lost", step=refinement_step, file_type="actions", content=json.dumps(actions, indent=4))

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
                # If outcome is None, continue executing remaining actions

            # After executing all refined actions, check if the game is completed without win/loss
            if not engine.won and not engine.lost:
                print(f"Level {level_id} completed without win/loss at refinement step {refinement_step}.")
                # Treat 'completed' as 'lost'
                self.save_file(level_id, status="lost", step=refinement_step, file_type="response", content=response)
                self.save_file(level_id, status="lost", step=refinement_step, file_type="actions", content=json.dumps(actions, indent=4))

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
                actions = eval(code_block)
                # Ensure only valid actions (right, left, up, down) are returned
                actions = [action for action in actions if action in ['right', 'left', 'up', 'down']]
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
