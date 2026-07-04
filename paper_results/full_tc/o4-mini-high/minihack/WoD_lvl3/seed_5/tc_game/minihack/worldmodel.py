from copy import deepcopy
from minihack_utils import directions

def sign(x):
    return 1 if x > 0 else -1 if x < 0 else 0

def transition_model(state, action):
    """
    state: a dict with keys like 
           'agent', 'minotaur', 'wall', 'an_uncursed_flint_stone',
           'inventory', 'won', 'lost', ...
    action: one of ['up','down',...,'pickup','apply','zap','select_f',
                    'shoot_up',...]
    """
    new_state = deepcopy(state)

    # ensure win/loss keys exist
    if 'won' not in new_state:
        new_state['won'] = False
    if 'lost' not in new_state:
        new_state['lost'] = False
    if 'inventory' not in new_state:
        new_state['inventory'] = []

    # local references
    agent_pos = new_state.get('agent', [[None, None]])[0]
    wall_positions = new_state.get('wall', [])
    mino_list = new_state.get('minotaur', [])
    # the wand‐on‐ground is actually called "an_uncursed_flint_stone"
    ground_wands = new_state.get('an_uncursed_flint_stone', [])

    ### 1) Agent action ###
    if action in directions:
        # Movement
        dx, dy = directions[action]
        tgt = [agent_pos[0] + dx, agent_pos[1] + dy]
        if tgt not in wall_positions:
            new_state['agent'] = [tgt]

    elif action == 'pickup':
        # pick up the wand if on ground at our feet
        if agent_pos in ground_wands:
            # remove from ground
            new_ground = [p for p in ground_wands if p != agent_pos]
            new_state['an_uncursed_flint_stone'] = new_ground
            # add to inventory
            if 'wand' not in new_state['inventory']:
                new_state['inventory'].append('wand')

    elif action == 'zap':
        # ready the wand if carried
        if 'wand' in new_state['inventory']:
            new_state['zap_ready'] = True

    elif action == 'select_f':
        # select the f-key after a zap
        if new_state.get('zap_ready') and 'wand' in new_state['inventory']:
            # become ready to fire
            new_state['f_selected'] = True
            # zap is consumed
            del new_state['zap_ready']

    elif action.startswith('shoot_'):
        # shoot only if f_selected
        if new_state.get('f_selected'):
            # fire a bolt in the given direction
            dir_name = action.split('_', 1)[1]
            dx, dy = directions.get(dir_name, [0, 0])
            bolt = agent_pos.copy()
            for _ in range(5):
                bolt[0] += dx
                bolt[1] += dy
                # stop at walls
                if bolt in wall_positions:
                    break
                # hit minotaur?
                if bolt in mino_list:
                    # kill it => win
                    new_state['minotaur'] = [m for m in mino_list if m != bolt]
                    new_state['won'] = True
                    break
            # clear the f_selected flag
            del new_state['f_selected']

    elif action == 'apply':
        # clear any in‐flight zap/f flags
        if 'zap_ready' in new_state:
            del new_state['zap_ready']
        if 'f_selected' in new_state:
            del new_state['f_selected']

    ### 2) Minotaur reaction ###
    # if still alive and no win yet, it may move/attack
    if not new_state['won'] and new_state.get('minotaur'):
        mpos = new_state['minotaur'][0]
        apos = new_state['agent'][0]
        dx = sign(apos[0] - mpos[0])
        dy = sign(apos[1] - mpos[1])
        # try to step closer on the larger axis first
        if abs(apos[0] - mpos[0]) >= abs(apos[1] - mpos[1]):
            cand = [mpos[0] + dx, mpos[1]]
            if cand not in wall_positions:
                new_state['minotaur'] = [cand]
            else:
                cand2 = [mpos[0], mpos[1] + dy]
                if cand2 not in wall_positions:
                    new_state['minotaur'] = [cand2]
        else:
            cand = [mpos[0], mpos[1] + dy]
            if cand not in wall_positions:
                new_state['minotaur'] = [cand]
            else:
                cand2 = [mpos[0] + dx, mpos[1]]
                if cand2 not in wall_positions:
                    new_state['minotaur'] = [cand2]

        # check for attack: now within Chebyshev distance ≤ 2
        m2 = new_state['minotaur'][0]
        a2 = new_state['agent'][0]
        if max(abs(m2[0] - a2[0]), abs(m2[1] - a2[1])) <= 2:
            new_state['lost'] = True

    return new_state