from utils import directions
import copy

def transition_model(state, action):
    """
    state: a dict with keys 'avatar', 'wall', 'floor', 'trap', 'goal'
    action: one of ['noop','left','right','up','down']
    returns a new state dict after applying the action
    """

    # deep‐copy so we don’t destroy the original
    new_state = copy.deepcopy(state)

    # read avatar position
    avatar_list = state.get('avatar', [])
    if not avatar_list:
        # no avatar, nothing to do
        return new_state

    # current avatar coords
    x, y = avatar_list[0]

    # compute displacement (noop falls back to [0,0])
    dx, dy = directions.get(action, [0, 0])
    new_pos = [x + dx, y + dy]

    # get obstacles and valid ground
    walls = state.get('wall', [])
    floor = state.get('floor', [])

    # only move if destination is on the floor and not a wall
    if new_pos in floor and new_pos not in walls:
        new_state['avatar'] = [new_pos]
    # else: leave avatar in place

    return new_state