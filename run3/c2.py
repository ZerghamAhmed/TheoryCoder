# make sure to include these import statements
from predicates import *
from copy import deepcopy
from utils import directions

def transition_model(state, action):
    # Make a deep copy of the state to avoid modifying the original
    next_state = deepcopy(state)

    # Define the agent that we control (Baba in this case)
    controllable = next_state['controllables'][0] if next_state['controllables'] else None
    if not controllable:
        # If there's no controllable object, the state does not change
        return next_state

    # Get the current position of the controllable object
    current_position = next_state[controllable][0]

    # Get the change in position based on the action taken
    movement = directions[action]
    new_position = [current_position[0] + movement[0], current_position[1] + movement[1]]

    # Check if the new position is a border or obstacle
    if new_position in next_state['border']:
        # If the move would place the object in a border, it is blocked, and no movement occurs
        return next_state

    # Check if the new position overlaps with a pushable object
    for pushable in next_state['pushables']:
        if new_position in next_state[pushable]:
            # If a pushable object is in the way, try to push it by moving it in the same direction
            new_pushable_position = [new_position[0] + movement[0], new_position[1] + movement[1]]
            if new_pushable_position in next_state['border'] or any(
                    new_pushable_position in next_state[other_obj] for other_obj in next_state['pushables']):
                # If the pushable object cannot be moved, movement is blocked
                return next_state
            else:
                # Move the pushable object to its new position
                next_state[pushable].remove(new_position)
                next_state[pushable].append(new_pushable_position)

    # Move the controllable object to its new position
    next_state[controllable][0] = new_position

    # Check if rules are being formed or broken
    formed_rules = []
    for potential_rule in generate_potential_rules(next_state):
        if rule_formed(next_state, *potential_rule):
            formed_rules.append(" ".join(potential_rule))

    # If the rule "baba_word is_word you_word" is broken, remove controllables
    if "baba_word is_word you_word" not in formed_rules:
        next_state['controllables'] = []

    # If no controllable object is left, set "lost" to True
    if not next_state['controllables']:
        next_state['lost'] = True

    # Return the next state
    return next_state
