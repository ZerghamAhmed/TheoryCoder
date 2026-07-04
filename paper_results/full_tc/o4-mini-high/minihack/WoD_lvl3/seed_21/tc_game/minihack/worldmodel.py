from copy import deepcopy
from minihack_utils import directions

def transition_model(state, action):
    """
    state: dict with keys like
      'agent': [[x,y]],
      'wall': [[x1,y1],...],
      'wand': [[wx,wy]]           (optional)
      'minotaur': [[mx,my]],
      'staircase_up': [[...]],
      'staircase_down': [[...]],
      'inventory': [...],
      'won': bool,
      'lost': bool,
      plus optional flags 'zapped', 'f_selected', 'applied'
    action: one of
      ['up','right','down','left',
       'up_right','down_right','down_left','up_left',
       'pickup','apply','zap','select_f',
       'shoot_up','shoot_right','shoot_down','shoot_left']
    """
    new_state = deepcopy(state)

    # === helper to clear shoot flags & remove wand from ground ===
    def clear_shoot_flags_and_wand():
        new_state.pop('zapped', None)
        new_state.pop('f_selected', None)
        # the real env actually consumes the wand on the floor when you shoot
        new_state.pop('wand', None)

    # normalize agent pos
    raw = new_state.get('agent', [0,0])
    if isinstance(raw, list) and raw and isinstance(raw[0], list):
        agent = raw[0].copy()
    else:
        agent = list(raw)

    walls = set(tuple(w) for w in new_state.get('wall', []))
    inv   = new_state.get('inventory', []).copy()

    # ================
    # Movement
    # ================
    if action in directions:
        dx, dy = directions[action]
        tgt = [agent[0] + dx, agent[1] + dy]
        if tuple(tgt) not in walls:
            agent = tgt

    # ================
    # pickup
    # ================
    elif action == 'pickup':
        if new_state.get('wand') and new_state['wand'][0] == agent:
            new_state.pop('wand', None)
            if 'wand' not in inv:
                inv.append('wand')
            new_state['inventory'] = inv

    # ================
    # apply (generic toggle)
    # ================
    elif action == 'apply':
        new_state['applied'] = True

    # ================
    # zap
    # ================
    elif action == 'zap':
        # only if wand in inventory
        if 'wand' in inv:
            new_state['zapped'] = True

    # ================
    # select_f
    # ================
    elif action == 'select_f':
        if new_state.get('zapped', False):
            new_state['f_selected'] = True

    # ================
    # shoot
    # ================
    elif action.startswith('shoot_'):
        dir_name = action[len('shoot_'):]
        dx, dy = directions.get(dir_name, [0,0])

        # only if we did zap & f_selected do we actually hit
        if new_state.get('zapped') and new_state.get('f_selected'):
            minos = new_state.get('minotaur', [])
            for step in range(1,6):
                probe = [agent[0] + dx*step, agent[1] + dy*step]
                if minos and probe == minos[0]:
                    new_state.pop('minotaur', None)
                    new_state['won'] = True
                    break

        # in all cases clear flags & remove ground-wand
        clear_shoot_flags_and_wand()

        # *** new: the agent also *moves* one square in the shot direction ***
        tgt = [agent[0] + dx, agent[1] + dy]
        if tuple(tgt) not in walls:
            agent = tgt

    # ================
    # unknown action
    # ================
    else:
        new_state['unknown_action'] = True

    # write back agent in nested form
    new_state['agent'] = [agent]
    return new_state