# make sure to include these import statements
from utils import directions

def transition_model(state, action):
    """
    This function takes the current state and the action and predicts the resulting state.
    """
    # Get the current position of the avatar
    avatar_pos = state.get('avatar')[0]

    # Deep copy the state to avoid modifying the input state
    new_state = {key: value.copy() if isinstance(value, list) else value for key, value in state.items()}
    new_avatar_pos = avatar_pos.copy()

    # Handle actions
    if action in directions:
        # Calculate the intended new position of the avatar
        movement = directions[action]
        new_avatar_pos[0] += movement[0]  # Update x-coordinate
        new_avatar_pos[1] += movement[1]  # Update y-coordinate

        # Check if the new position is valid (i.e., not a wall)
        if new_avatar_pos not in state.get('wall', []):
            # Update the avatar's position in the new state
            new_state['avatar'] = [new_avatar_pos]
        else:
            # Movement is blocked by a wall; avatar stays in the same position
            new_avatar_pos = avatar_pos
    elif action == 'noop':
        # "No operation" means the avatar doesn't move
        new_avatar_pos = avatar_pos

    # Return the potentially modified state
    return new_state