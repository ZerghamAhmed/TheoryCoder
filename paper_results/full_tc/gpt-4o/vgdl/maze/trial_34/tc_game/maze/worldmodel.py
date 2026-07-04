# make sure to include these import statements
from utils import directions

def transition_model(state, action):
    """
    Given a state and an action, return the next state based on the provided transition rules.
    """
    # Create a deep copy of the current state so we don't modify the original one
    new_state = {
        key: state.get(key, [])[:] if isinstance(state.get(key, []), list) else state.get(key)
        for key in state
    }
    
    # Get the avatar's current position
    avatar_pos = state.get('avatar', [[0, 0]])[0]

    # Determine the effect of the action
    if action in directions:
        dx, dy = directions[action]
        new_avatar_pos = [avatar_pos[0] + dx, avatar_pos[1] + dy]

        # Check if the new position is valid (not blocked by a wall)
        if new_avatar_pos in state.get('floor', []) and new_avatar_pos not in state.get('wall', []):
            # Update the avatar's position
            new_state['avatar'] = [new_avatar_pos]
        else:
            # New position is invalid; avatar stays in the same position
            new_state['avatar'] = [avatar_pos]
    
    elif action == 'noop':
        # No-op means no movement, so the state remains unchanged
        new_state['avatar'] = [avatar_pos]
    
    return new_state