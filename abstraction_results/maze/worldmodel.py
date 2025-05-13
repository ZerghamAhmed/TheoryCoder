# make sure to include these import statements
from copy import deepcopy
from utils import directions

def transition_model(state, action):
    # Make a deep copy of the state to avoid modifying the original state
    new_state = deepcopy(state)

    # Extract the current position of the rat
    rat_position = new_state['rat'][0]

    # If the action is 'noop', return the state as is
    if action == 'noop':
        return new_state

    # Calculate the new position based on the action
    move = directions.get(action)
    if move is not None:
        new_position = [rat_position[0] + move[0], rat_position[1] + move[1]]
        
        # Check if the new position is within the bounds and not a wall
        if new_position in new_state['floor'] and new_position not in new_state['wall']:
            new_state['rat'] = [new_position]

    return new_state
