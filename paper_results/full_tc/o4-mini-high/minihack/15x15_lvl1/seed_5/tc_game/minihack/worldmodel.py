from copy import deepcopy
from minihack_utils import directions

def transition_model(state, action):
    """
    state: a dict, e.g.
      {
        'staircase_up': [[36,11]],
        'human_rogue_called_agent': [[38,8]],
        'staircase_down': [[40,7]],
        'inventory': [],
        'won': False,
        'lost': False
      }
    action: one of ['up','down','left','right','up_right',...]
    """
    # deep copy so we don't mutate the original
    new_state = deepcopy(state)
    
    # fetch the current agent position
    # state.get(...) to avoid KeyError if ever missing
    agent_pos_list = new_state.get('human_rogue_called_agent')
    if agent_pos_list is None or len(agent_pos_list)==0:
        # no agent in state? just return unchanged
        return new_state

    x, y = agent_pos_list[0]
    
    # get the move delta
    dx, dy = directions.get(action, (0,0))
    
    # apply the move
    new_x = x + dx
    new_y = y + dy
    
    # write back the new position as a singleton list
    new_state['human_rogue_called_agent'] = [[new_x, new_y]]
    
    # (optionally) detect if we've stepped onto the downstairs
    # and mark a win or level‐complete. Uncomment if desired:
    #
    # down = new_state.get('staircase_down', [])
    # if down and [new_x, new_y] == down[0]:
    #     new_state['won'] = True
    
    return new_state