from copy import deepcopy
from utils import directions

def transition_model(state, action):
    # Create a deepcopy of the state to avoid modifying the original state directly
    new_state = deepcopy(state)
    
    # Get the action direction vector
    direction = directions[action]
    
    # Iterate over all controllable objects and attempt to move them
    for controllable in state['controllables']:
        # Get the current position of the controllable object
        current_position = state[controllable][0]
        
        # Calculate the new position after the action
        new_position = [current_position[0] + direction[0], current_position[1] + direction[1]]
        
        # Check if the new position is within the borders
        if new_position in state['border']:
            continue
        
        # Check if the new position contains a pushable object
        can_move = True
        for pushable in state['pushables']:
            if new_position in state[pushable]:
                # Try to push the pushable object
                if not push_pushable(state, new_state, pushable, new_position, direction):
                    can_move = False
                    break
        
        # If the move is valid, update the position of the controllable
        if can_move:
            new_state[controllable][0] = new_position
    
    return new_state

def push_pushable(state, new_state, pushable, position, direction):
    # Calculate the new position of the pushable object
    next_position = [position[0] + direction[0], position[1] + direction[1]]
    
    # Check if next position is within the borders
    if next_position in state['border']:
        return False
    
    # Check if the next position has another pushable object
    for other_pushable in state['pushables']:
        if next_position in state[other_pushable]:
            # Recursively try to push the next pushable
            if not push_pushable(state, new_state, other_pushable, next_position, direction):
                return False
    
    # If the next position is valid, move the pushable
    for pushable in state['pushables']:
        if position in state[pushable]:
            new_state[pushable].remove(position)
            new_state[pushable].append(next_position)
    
    return True