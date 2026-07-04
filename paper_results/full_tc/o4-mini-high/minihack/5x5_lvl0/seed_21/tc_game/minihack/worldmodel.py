from minihack_utils import directions

def transition_model(state, action):
    """
    A deterministic geometric transition model for the human_rogue_called_agent avatar.
    We assume:
     - staircase_up, staircase_down, inventory, won, lost are static under movement
     - human_rogue_called_agent moves by one step in `directions[action]`
     - if you later want to model obstacles, insert a collision check before updating pos
    
    state: dict with keys e.g. 'human_rogue_called_agent', 'staircase_up', 'staircase_down', 'inventory', 'won', 'lost'
    action: one of ['up','down','left','right','up_right','up_left','down_right','down_left']
    returns: new_state dict
    """
    # Copy over all existing fields (shallow copy)
    new_state = state.copy()

    # Read out the current agent position (a list of one [x,y] pair)
    agent_list = state.get('human_rogue_called_agent', [])
    if not agent_list:
        # no agent in state? just return unchanged
        return new_state

    x, y = agent_list[0]

    # Look up the direction delta
    dx, dy = directions.get(action, [0, 0])

    # Compute new candidate position
    new_x, new_y = x + dx, y + dy

    # Here you could add a collision check:
    # if is_blocked((new_x,new_y)):
    #     new_x, new_y = x, y

    # Update the agent’s position in the next state
    new_state['human_rogue_called_agent'] = [[new_x, new_y]]

    # Everything else (stairs, inventory, win/lose flags) remains unchanged
    return new_state