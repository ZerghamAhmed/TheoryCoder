# make sure to include these import statements
from copy import deepcopy
from utils import directions

def transition_model(state, action):
    # First, we need to copy the current state to avoid mutating the original one
    new_state = deepcopy(state)
    
    # Get the position of the controllable object, which in this case is "baba_obj"
    controllable_obj = new_state['controllables'][0]  # 'baba_obj'
    current_position = new_state[controllable_obj][0]  # We assume only one baba_obj, for now

    # Determine the direction of movement from the action
    move = directions[action]
    
    # Calculate the new position after the action
    new_position = [current_position[0] + move[0], current_position[1] + move[1]]

    # Check if the new position is within borders or blocked by any immovable objects
    if new_position in new_state['border']:
        # If the new position is a border, movement is prevented
        return new_state  # No change in state

    # Check if the new position is overlapped by an "overlappable" object like the flag
    if new_position in new_state['overlappables']:
        # Movement is allowed but baba overlaps with another object like the flag
        new_state[controllable_obj][0] = new_position

    # Check if the new position collides with a pushable object (e.g., "is_word", "flag_word")
    elif any(new_position in new_state[pushable] for pushable in new_state['pushables']):
        # Determine if the object in the new position can be pushed
        for pushable in new_state['pushables']:
            if new_position in new_state[pushable]:
                # Calculate the position to which the pushable object would move
                new_pushable_position = [new_position[0] + move[0], new_position[1] + move[1]]
                
                # If the new pushable position is valid (not blocked by borders or other objects)
                if new_pushable_position not in new_state['border'] and \
                   new_pushable_position not in new_state['pushables']:
                    # Push the object to the new position
                    new_state[pushable][new_state[pushable].index(new_position)] = new_pushable_position
                    # Move baba_obj to the new position
                    new_state[controllable_obj][0] = new_position
                else:
                    # If the pushable object cannot move, baba stays in the same position
                    return new_state
    else:
        # No blockages, move baba_obj to the new position
        new_state[controllable_obj][0] = new_position
    
    # Return the updated state
    return new_state
