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
            # If no obstacles, move the entity
            new_state[entity][0] = new_pos

            # Check for interactions (e.g., if new position coincides with a word or object)
            for obj_key in state.keys():
                if obj_key.endswith('_word') or obj_key.endswith('_obj'):
                    if new_pos in state[obj_key]:
                        # Handle specific collision behavior
                        if obj_key == 'flag_word':
                            # Example behavior: flag_word moves when baba_obj collides
                            new_state[obj_key][0] = [new_pos[0], new_pos[1] + 1]

        else:
            # Movement is blocked, no changes to position
            new_state[entity][0] = current_pos

    return new_state
