from utils import directions

def transition_model(state, action):
    """
    Given the current state and an action, returns the next state.
    Movement is blocked by walls; 'noop' or blocked moves leave the avatar in place.
    """
    # Extract static elements with explicit defaults
    walls           = state.get('wall', [])
    floors          = state.get('floor', [])
    traps           = state.get('trap', [])
    goals           = state.get('goal', [])
    avatar_positions = state.get('avatar', [])

    # Prepare new avatar list (will be one-element list or empty if no avatar)
    new_avatar_positions = []

    if avatar_positions:
        # Current avatar position (assume single avatar)
        old_x, old_y = avatar_positions[0]

        # Lookup movement delta; default to [0,0] so 'noop' or unknown keeps position
        dx, dy = directions.get(action, [0, 0])

        # Compute proposed new position
        new_x = old_x + dx
        new_y = old_y + dy
        proposed_pos = [new_x, new_y]

        # If the proposed position is blocked by a wall, stay in place
        if proposed_pos in walls:
            proposed_pos = [old_x, old_y]

        new_avatar_positions = [proposed_pos]

    # Reconstruct the next state dictionary
    next_state = {
        'wall': walls,
        'floor': floors,
        'trap': traps,
        'goal': goals,
        'avatar': new_avatar_positions
    }

    return next_state