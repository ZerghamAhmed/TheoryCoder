# make sure to include these import statements
from predicates import *
from copy import deepcopy
from games import BabaIsYou
from babareport import BabaReportUpdater
from utils import directions

def transition_model(state, action):
    # Copy the current state to avoid modifying it directly
    new_state = deepcopy(state)

    # Get the list of controllable objects (entities the player can control)
    controllables = state['controllables']

    # Get movement direction from the action
    direction = directions[action]

    # Apply movement to controllable objects
    for entity in controllables:
        # Get current position of the entity
        current_pos = new_state[entity][0]  # assuming controllables are single-instance
        new_pos = [current_pos[0] + direction[0], current_pos[1] + direction[1]]

        # Check if the new position is a border or other immovable object
        if new_pos not in state['border']:
            # Check for interactions with other objects or words (collision-based interactions)
            for obj in ['baba_word', 'is_word', 'you_word', 'flag_word', 'flag_obj', 'win_word']:
                if new_pos in state[obj]:
                    # If the object is a word, modify its position (collision effect)
                    if '_word' in obj:
                        # Move the word object to the next available spot based on action direction
                        new_state[obj][0] = [new_state[obj][0][0] + direction[0], new_state[obj][0][1] + direction[1]]
                    # If flag object is collided with, you win the game (update win state)
                    if obj == 'flag_obj':
                        new_state['won'] = True
                    break

            # If no obstacles or special interactions, move the controllable object
            new_state[entity][0] = new_pos
        else:
            # Movement is blocked, no changes to position
            new_state[entity][0] = current_pos

    return new_state
