from copy import deepcopy
from minihack_utils import directions

def is_blocked(pos, state):
    """
    Return True if pos is in the state's blocked_cells list.
    """
    blocked = state.get('blocked_cells', [])
    # make sure we compare lists of ints
    return pos in blocked

def is_trap(pos, state):
    """
    Return True if pos is in the state's trap_cells list.
    """
    traps = state.get('trap_cells', [])
    return pos in traps

def transition_model(state, action):
    """
    state: dict with keys like 'human_rogue_called_agent', 'staircase_up', 'staircase_down',
           'inventory', 'won', 'lost', and optionally 'blocked_cells', 'trap_cells'
    action: one of ['up','down','left','right','up_left',…]
    returns: next_state (dict)
    """
    new_state = deepcopy(state)

    # 1) get current agent position
    agent_list = new_state.get('human_rogue_called_agent', [])
    if not agent_list:
        # nothing to do
        return new_state
    old_pos = agent_list[0]

    # 2) compute candidate new position
    dx, dy = directions.get(action, (0, 0))
    cand_pos = [old_pos[0] + dx, old_pos[1] + dy]

    # 3) collision check
    if is_blocked(cand_pos, state):
        new_pos = old_pos
    else:
        new_pos = cand_pos

    # 4) trap check: stepping on a trap kills you
    if is_trap(cand_pos, state):
        new_state['lost'] = True

    # 5) write back the agent's position
    new_state['human_rogue_called_agent'] = [new_pos]

    # other keys ('staircase_up', 'staircase_down', 'inventory', 'won', etc.) stay as is
    return new_state