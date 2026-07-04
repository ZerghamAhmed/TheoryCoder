from utils import directions

def transition_model(state, action):
    """
    Given a `state` dict and an `action` string, returns the next state dict.
    Uses explicit defaults for all .get() calls, and blocks movement into walls.
    """
    # pull out all the pieces, with safe defaults
    walls  = state.get('wall', [])
    floor  = state.get('floor', [])
    trap   = state.get('trap', [])
    goal   = state.get('goal', [])
    avatar = state.get('avatar', [])  # expect [[x,y]] or []

    # figure out the delta for this action (noop -> [0,0])
    dx, dy = directions.get(action, [0, 0])

    # current avatar position
    if avatar:
        x, y = avatar[0]
    else:
        # if there's no avatar key or it's empty, assume (0,0)
        x, y = 0, 0

    # candidate new position
    new_pos = [x + dx, y + dy]

    # if the new position is a wall, stay in place
    if new_pos in walls:
        new_avatar = [[x, y]]
    else:
        new_avatar = [new_pos]

    # build and return the next state
    return {
        'wall':  walls,
        'floor': floor,
        'trap':  trap,
        'goal':  goal,
        'avatar': new_avatar
    }