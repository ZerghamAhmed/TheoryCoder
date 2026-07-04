from copy import deepcopy
from utils import directions

def transition_model(state, action):
    """
    Given a `state` dict and an `action` string, returns the next state dict.
    We deep-copy the incoming state so that arbitrary keys (e.g. 'cheese' or 'goal')
    are preserved verbatim, and then only update the 'avatar' field.
    """
    # Copy everything so we don't accidentally lose 'cheese', 'goal', 'trap', etc.
    new_state = deepcopy(state)

    walls  = state.get('wall', [])
    avatar = state.get('avatar', [])

    # If there's no avatar entry, assume it's at (0,0)
    if avatar:
        x, y = avatar[0]
    else:
        x, y = 0, 0

    # How much to move for this action
    dx, dy = directions.get(action, [0, 0])
    candidate = [x + dx, y + dy]

    # Block movement into walls
    if candidate in walls:
        new_avatar = [[x, y]]
    else:
        new_avatar = [candidate]

    new_state['avatar'] = new_avatar
    return new_state