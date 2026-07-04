# make sure to include these import statements
from copy import deepcopy
from utils import directions

def transition_model(state, action):
    """
    Given a state and an action, return the next state based on the provided transition rules.
    """
    # Create a deep copy of the current state so we don't modify the original one
    new_state = {
        key: deepcopy(state.get(key, [])) if isinstance(state.get(key, []), list) else state.get(key, [])
        for key in state
    }

    # Get the avatar's current position and other entity lists
    avatar_pos = state.get('avatar', [[0, 0]])[0]
    boxes = state.get('box', [])
    walls = state.get('wall', [])
    floor = state.get('floor', [])

    # If no movement action, return the state as is
    if action == 'noop':
        return new_state

    # Determine the effect of the action
    dx, dy = directions.get(action, [0, 0])
    new_avatar_pos = [avatar_pos[0] + dx, avatar_pos[1] + dy]

    # Case 1: If new avatar position is invalid (not in floor or blocked by wall)
    if new_avatar_pos not in floor or new_avatar_pos in walls:
        return new_state  # Avatar stays in its current location

    # Case 2: If new avatar position contains a box
    if new_avatar_pos in boxes:
        # Calculate the new position for the box
        box_index = boxes.index(new_avatar_pos)
        new_box_pos = [new_avatar_pos[0] + dx, new_avatar_pos[1] + dy]

        # Check if the box can be moved (new box position must be in floor, not in walls or other boxes)
        if new_box_pos in floor and new_box_pos not in walls and new_box_pos not in boxes:
            # Move the box and the avatar
            new_state['box'][box_index] = new_box_pos
            new_state['avatar'] = [new_avatar_pos]
        else:
            # The box can't be moved, so the avatar also doesn't move
            return new_state
    else:
        # Case 3: The new avatar position is valid and doesn't contain a box
        new_state['avatar'] = [new_avatar_pos]

    # Return the modified state
    return new_state