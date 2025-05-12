# make sure to include these import statements
from utils import directions

def transition_model(state, action):
    # Get the current position of the avatar
    avatar_pos = state.get('avatar', [[0, 0]])[0]
    walls = state.get('wall', [])
    
    # Determine the new position based on the action
    if action in directions:
        direction = directions[action]
        new_pos = [avatar_pos[0] + direction[0], avatar_pos[1] + direction[1]]
    else:
        # If the action is 'noop', the position doesn't change
        new_pos = avatar_pos

    # Check if the new position is valid (i.e., not a wall and within bounds)
    if new_pos not in walls:
        # Update the avatar's position if the move is valid
        new_state = state.copy()
        new_state['avatar'] = [new_pos]
    else:
        # If the move is invalid, stay in the current position
        new_state = state

    return new_state
