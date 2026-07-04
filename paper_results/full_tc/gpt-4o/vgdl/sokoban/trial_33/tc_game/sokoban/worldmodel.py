# make sure to include these import statements
from copy import deepcopy
from utils import directions

def transition_model(state, action):
    """
    Transition model for predicting the next state based on the current state and action.

    Parameters:
    state (dict): Current state of the game. Contains entities like 'avatar', 'wall', 'floor', etc.
    action (str): The action to be executed. ('noop', 'right', 'left', 'up', 'down')

    Returns:
    dict: The predicted next state after executing the action.
    """
    # Deep copy the current state to avoid modifying the input
    next_state = deepcopy(state)
    
    # Get the current position of the avatar
    avatar_pos = state.get('avatar', [])[0]  # Assume single avatar in the first position
    if not avatar_pos:
        return next_state  # If the avatar position is not valid, return an unchanged state

    # If action is noop, return the same state
    if action == 'noop':
        return next_state

    # Calculate the new position of the avatar based on the action
    direction = directions.get(action, [0, 0])
    new_avatar_pos = [avatar_pos[0] + direction[0], avatar_pos[1] + direction[1]]

    # Retrieve current state information
    walls = state.get('wall', [])
    floor = state.get('floor', [])
    boxes = state.get('box', [])

    # If the new position is a wall or not in the floor, the avatar cannot move
    if new_avatar_pos in walls or new_avatar_pos not in floor:
        return next_state

    # Check if the new position is occupied by a box
    if new_avatar_pos in boxes:
        # Calculate the new position for the box being pushed
        new_box_pos = [new_avatar_pos[0] + direction[0], new_avatar_pos[1] + direction[1]]

        # The box can only be pushed if the next position is within bounds (on the floor) 
        # and not colliding with walls or other boxes
        if new_box_pos in walls or new_box_pos not in floor or new_box_pos in boxes:
            return next_state  # If the box cannot move, neither the avatar nor the box moves

        # Update the box's position
        next_state['box'].remove(new_avatar_pos)
        next_state['box'].append(new_box_pos)

    # Update the avatar's position
    next_state['avatar'] = [new_avatar_pos]

    # Return the updated state
    return next_state