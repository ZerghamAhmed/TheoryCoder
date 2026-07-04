# make sure to include these import statements
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
    next_state = state.copy()
    
    # Get the current position of the avatar
    avatar_pos = state.get('avatar', [])[0]  # Get the first element since it is a list of lists
    if not avatar_pos:
        return next_state  # If the avatar position is not valid, return an unchanged state

    # If action is noop, return the same state
    if action == 'noop':
        return next_state

    # Calculate the new position of the avatar based on the action
    direction = directions.get(action, [0, 0])
    new_avatar_pos = [avatar_pos[0] + direction[0], avatar_pos[1] + direction[1]]

    # Check if the new position is valid (not colliding with walls and on the floor)
    walls = state.get('wall', [])
    floor = state.get('floor', [])

    # If the new position is not in walls and is on the floor, update avatar's position
    if new_avatar_pos not in walls and new_avatar_pos in floor:
        next_state['avatar'] = [new_avatar_pos]  # Update the avatar's position

    # Return the updated state
    return next_state