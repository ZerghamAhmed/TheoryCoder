from copy import deepcopy
from utils import directions

def transition_model(state, action):
    """
    Given the current state and an action, returns the next state.
    Movement is blocked by walls; 'noop' or blocked moves leave the avatar in place.
    The cheese (goal) is preserved in the returned state.
    """
    # Make a full copy so we don’t clobber the original
    new_state = deepcopy(state)

    # Static elements
    walls   = new_state.get('wall', [])
    floors  = new_state.get('floor', [])
    traps   = new_state.get('trap', [])
    cheese  = new_state.get('cheese', [])  # the goal positions

    # Current avatar
    avatar_positions = new_state.get('avatar', [])
    new_avatar_positions = []

    if avatar_positions:
        old_x, old_y = avatar_positions[0]
        dx, dy = directions.get(action, [0, 0])

        # Compute where we’d like to go
        new_x, new_y = old_x + dx, old_y + dy
        proposed = [new_x, new_y]

        # If there’s a wall, we stay put
        if proposed in walls:
            proposed = [old_x, old_y]

        new_avatar_positions = [proposed]

    # Write back the avatar (empty list if none)
    new_state['avatar'] = new_avatar_positions

    # Nothing else changes—cheese, floor, traps, walls stay the same
    return new_state