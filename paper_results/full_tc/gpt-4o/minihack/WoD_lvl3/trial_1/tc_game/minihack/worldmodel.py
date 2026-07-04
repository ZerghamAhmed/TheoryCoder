# make sure to include these import statements
from copy import deepcopy
from minihack_utils import directions

def transition_model(state, action):
    new_state = deepcopy(state)  # Use deepcopy to ensure a full copy of the state
    agent_pos = new_state.get('agent', [None])[0]
    wand_pos = new_state.get('wand', [[None, None]])[0]
    minotaur_pos = new_state.get('minotaur', [[None, None]])[0]
    inventory = new_state.get('inventory', [])
    won = new_state.get('won')
    lost = new_state.get('lost')

    # Safety check for wall and lost condition if minotaur position or agent position are not defined
    if agent_pos is None or minotaur_pos is None:
        return new_state

    # Movement actions
    if action in directions:
        move = directions[action]
        new_pos = [agent_pos[0] + move[0], agent_pos[1] + move[1]]

        # Check if the new position is within bounds and not a wall
        if new_pos not in new_state.get('wall', []):
            new_state['agent'] = [new_pos]

    # Pickup action
    elif action == 'pickup':
        # Pickup the wand if the agent is on the wand's position
        if agent_pos == wand_pos and "wand" not in inventory:
            new_state['inventory'].append('wand')

    # Zap action
    elif action == 'zap':
        # Set a flag for zap initiation
        new_state['zap'] = True

    # Select 'f' action (must come after zapping)
    elif action == 'select_f':
        if new_state.get('zap'):
            new_state['select_f'] = True

    # Shooting actions (primitive, follows zap and select_f)
    elif action.startswith('shoot_'):
        if 'wand' in inventory and new_state.get('zap') and new_state.get('select_f'):
            direction = action.split('_')[1]
            move = directions.get(direction)

            if not move:
                return new_state  # in case there's an invalid shoot direction
            
            # Calculate the firing path
            for i in range(1, 6):  # 5 squares in the intended direction
                shot_pos = [agent_pos[0] + i * move[0], agent_pos[1] + i * move[1]]

                # Check if it hits the minotaur
                if shot_pos == minotaur_pos:
                    new_state['won'] = True
                    new_state['minotaur'] = []  # Minotaur dies
                    break

            # Reset zap and select_f as they are single-use flags per shot
            new_state['zap'] = False
            new_state['select_f'] = False

    # Minotaur movement
    # Move the minotaur closer to the agent every 4 actions
    if 'move_count' not in new_state:
        new_state['move_count'] = 0
    new_state['move_count'] = (new_state['move_count'] + 1) % 4

    if new_state['move_count'] == 0:
        if agent_pos[0] > minotaur_pos[0]:
            minotaur_pos[0] += 1
        elif agent_pos[0] < minotaur_pos[0]:
            minotaur_pos[0] -= 1
        elif agent_pos[1] > minotaur_pos[1]:
            minotaur_pos[1] += 1
        elif agent_pos[1] < minotaur_pos[1]:
            minotaur_pos[1] -= 1

        new_state['minotaur'] = [minotaur_pos]

    # Ensure the loss condition is checked
    # If the minotaur reaches the agent, the game is lost
    if (agent_pos[0] == minotaur_pos[0] and abs(agent_pos[1] - minotaur_pos[1]) <= 1) or \
       (agent_pos[1] == minotaur_pos[1] and abs(agent_pos[0] - minotaur_pos[0]) <= 1):
        new_state['lost'] = True

    return new_state