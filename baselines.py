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
    def __init__(self, episode_length, world_model_load_name, domain_file_name, refine=False):
        self.episode_length = episode_length
        self.world_model_load_name = world_model_load_name
        self.domain_file_name = domain_file_name
        self.refine_enabled = refine  # Control refinement

        self.language_model = 'gpt-4o'  # You can change to your preferred model
        # self.language_model='gpt-3.5-turbo',
        chat = ChatOpenAI(model_name=self.language_model, temperature=1.0)
        self.query_lm = lambda prompt: chat(prompt.to_messages()).content

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

    def _make_langchain_prompt(self, text, **kwargs):
        """Make the Langchain prompt, the same way as in tbrl.py."""
        x = HumanMessagePromptTemplate.from_template(text)
        chat_prompt = ChatPromptTemplate.from_messages([x])
        prompt = chat_prompt.format_prompt(**kwargs)
        return prompt

    def query_lm(self, prompt):
        """Query the LLM using the formatted prompt and save the response."""
        response = self.chat.invoke([prompt]).content  # Use invoke instead of call
        self.save_response(response, "baseline_response.txt")  # Save to txt file
        return response

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

    def initial_request_prompt(self, state):
        """Generate the initial request prompt for the LLM using _make_langchain_prompt."""
        domain_content = self.load_file(self.domain_file_name)
        world_model_content = self.load_file(self.world_model_load_name)

        prompt = self._make_langchain_prompt(INITIAL_REQUEST_PROMPT,  # Same method as tbrl.py
            **{
            'actions_set': self.actions_set,
            'state_format': engine.state_format,  # Fix: Passing state directly
            'current_state': state,
            'domain_file': domain_content,
            'world_model': world_model_content,
            'utils' : "directions = {\n    'left': [-1, 0],\n    'right': [1, 0],\n    'up': [0, 1],\n    'down': [0, -1],\n}"
            }
        )

        return prompt

    

    def refine_prompt(self, state, previous_actions, replay_buffer):
        """Generate the refine request prompt for the LLM based on failed actions."""
        domain_content = self.load_file(self.domain_file_name)
        world_model_content = self.load_file(self.world_model_load_name)

        prompt_text = REFINE_PROMPT

        prompt = self._make_langchain_prompt(prompt_text,  # Using _make_langchain_prompt
            actions_set=self.actions_set,
            state=engine.state_format,  # Fix: Passing state directly
            replay_buffer=replay_buffer,
            domain_file=domain_content,
            world_model=world_model_content,
            utils="directions = {\n    'left': [-1, 0],\n    'right': [1, 0],\n    'up': [0, 1],\n    'down': [0, -1],\n}"
        )
        return prompt

    def step_env(self, engine, action):
        """Step the game engine, store the transition in the replay buffer."""
        # Get the current state before the action
        state = deepcopy(engine.get_obs())

        # Step the environment by executing the action
        engine.step(action)

        # Get the next state after the action
        next_state = deepcopy(engine.get_obs())

        # Store the transition in the replay buffer
        self.replay_buffers.append((state, action, next_state))

        # Check for win/loss
        if engine.won:
            print("Agent won!")
            return True  # Stop the loop
        elif engine.lost:
            print("Agent lost.")
            return False  # Stop the loop
        return None  # Continue the loop if neither win nor loss

    def run(self, engine):
        """Run the initial request to get action sequence from LLM and evaluate it."""
        state = self.format_state(deepcopy(engine.get_obs()))  # Format the state

        # Initial request to get the action sequence
        prompt = self.initial_request_prompt(state)
        print(f"Sending initial request: {prompt}")
        response = self.query_lm(prompt)

        # Extract action set from response
        actions = self.extract_actions(response)

        # Run the actions through the game engine
        for action in actions:
            result = self.step_env(engine, action)

            # Check for win/loss after each action
            if result is True:
                self.save_actions(actions, "baseline")
                return True
            elif result is False:
                self.save_actions(actions, "baseline")
                if self.refine_enabled:
                    self.refine(engine, actions)
                else:
                    return False

    def refine(self, engine, previous_actions):
        """Refine actions if agent loses or fails to win the game."""
        print("Refinement enabled. Refining...")
        # Send refinement request with replay buffer information
        replay_buffer = self.replay_buffers
        prompt = self.refine_prompt(engine.get_obs(), previous_actions, replay_buffer)
        print(f"Sending refinement request: {prompt}")
        response = self.query_lm(prompt)

        # Extract refined actions and re-run the game
        actions = self.extract_actions(response)
        self.run(engine)

    def extract_actions(self, response):
        """Extract the action list from the LLM response."""
        try:
            # Use regular expression to find Python code block
            code_block_match = re.findall(r'```Python([\s\S]*?)```', response)
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
            return []  # or you can raise the exception if you prefer

    def save_response(self, response, file_name):
        """Save LLM response to a file."""
        with open(file_name, 'w') as f:
            f.write(response)

    def save_actions(self, actions, result):
        """Save the action set to a file after winning or losing."""
        file_name = f"{result}_actions.txt"
        with open(file_name, 'w') as f:
            f.write(f"Actions: {actions}\n")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, default='baba')
    parser.add_argument('--levels', type=str, default="[('demo_LEVELS', 0)]")
    parser.add_argument('--episode-length', type=int, default=20)
    parser.add_argument('--world-model-file-name', type=str, default='worldmodel.py')
    parser.add_argument('--domain-file-name', type=str, default='domain.pddl')
    parser.add_argument('--predicates-file-name', type=str, default='predicates')
    parser.add_argument('--json-reporter-path', type=str, default='KekeCompetition-main/Keke_JS/reports/TBRL_BABA_REPORT.json')
    parser.add_argument('--refine', action='store_true', help="Enable refinement if the LLM's action sequence leads to a loss")

    args = parser.parse_args()
    levels = eval(args.levels)

    for level_set, level_id in levels:
        if args.game == 'baba':
            engine = BabaIsYou(level_set=level_set, level_id=level_id)

        agent = Baselines(
            episode_length=args.episode_length,
            world_model_load_name=args.world_model_file_name,
            domain_file_name=args.domain_file_name,
            refine=args.refine
        )
        agent.run(engine)