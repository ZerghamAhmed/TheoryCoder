from minihack_utils import directions
import copy

def transition_model(state, action):
    """
    state: dict with keys like 'agent', 'minotaur', 'wall', 'inventory', 'won', 'lost'
    action: one of ['up','down','left','right','up_left','up_right','down_left','down_right',
                    'pickup','apply','zap','select_f',
                    'shoot_up','shoot_down','shoot_left','shoot_right']
    returns a new state dict (deep copy) after applying the action and moving the minotaur
    """
    new_state = copy.deepcopy(state)
    # extract positions and simple fields
    agent_list = new_state.get('agent', [])
    agent_pos = agent_list[0] if agent_list else [0,0]
    mino_list = new_state.get('minotaur', [])
    has_minotaur = (len(mino_list) > 0)
    mino_pos = mino_list[0] if has_minotaur else None
    walls = new_state.get('wall', [])
    inv = new_state.get('inventory', [])
    
    # bounded modulo counter for minotaur movement
    turn_mod = new_state.get('turn_mod', 0)
    # bounded wand use phase: 0 = nothing, 1 = after zap, 2 = after select_f
    wand_phase = new_state.get('wand_phase', 0)
    
    # 1) Apply the agent action
    if action in directions:
        # movement attempt
        dx, dy = directions[action]
        cand = [agent_pos[0] + dx, agent_pos[1] + dy]
        if cand not in walls:
            new_state['agent'] = [cand]
        # walking does not auto-pickup
        # wand_phase unchanged
        
    elif action == 'pickup':
        # if there's a wand on the ground where the agent stands
        wand_on_ground = new_state.get('wand_pos', None)
        if wand_on_ground and wand_on_ground == agent_pos:
            # pick it up
            if 'wand' not in inv:
                inv.append('wand')
                new_state['inventory'] = inv
            # remove the ground item
            del new_state['wand_pos']
        # wand_phase unchanged
        
    elif action == 'zap':
        # begin wand use
        if 'wand' in inv:
            wand_phase = 1
        else:
            # still change something
            wand_phase = 0
    
    elif action == 'select_f':
        # select the f key after a zap
        if wand_phase == 1:
            wand_phase = 2
        else:
            # invalid sequence, reset
            wand_phase = 0
    
    elif action == 'apply':
        # we interpret apply as canceling any partial wand use
        wand_phase = 0
    
    elif action.startswith('shoot_'):
        # shooting direction
        # must have done zap + select_f
        dir_name = action.split('_',1)[1]
        vec = directions.get(dir_name, [0,0])
        if wand_phase == 2 and has_minotaur:
            # trace bolt up to 5 squares
            for i in range(1,6):
                targ = [agent_pos[0] + vec[0]*i,
                        agent_pos[1] + vec[1]*i]
                if targ == mino_pos:
                    # hit!
                    new_state['minotaur'] = []
                    new_state['won'] = True
                    break
        # after any shoot, reset wand_phase
        wand_phase = 0
    
    else:
        # unexpected action, do nothing but still update minotaur_phase
        pass
    
    # 2) Update wand_phase and turn_mod
    new_state['wand_phase'] = wand_phase
    turn_mod = (turn_mod + 1) % 5
    new_state['turn_mod'] = turn_mod
    
    # 3) Move the minotaur every 5 agent actions (when turn_mod==0)
    if has_minotaur and not new_state.get('won', False) and not new_state.get('lost', False):
        if turn_mod == 0:
            ax, ay = new_state['agent'][0]
            mx, my = mino_pos
            dx = ax - mx
            dy = ay - my
            # sign function
            sx = 0 if dx==0 else (1 if dx>0 else -1)
            sy = 0 if dy==0 else (1 if dy>0 else -1)
            # try diagonal move first
            cand = [mx + sx, my + sy]
            if cand in walls:
                # try horizontal
                cand = [mx + sx, my]
                if cand in walls:
                    # try vertical
                    cand = [mx, my + sy]
                    if cand in walls:
                        # blocked
                        cand = [mx, my]
            # apply move
            new_state['minotaur'] = [cand]
            # check adjacency -> loss
            ax, ay = new_state['agent'][0]
            if max(abs(cand[0]-ax), abs(cand[1]-ay)) <= 1:
                new_state['lost'] = True
    
    return new_state