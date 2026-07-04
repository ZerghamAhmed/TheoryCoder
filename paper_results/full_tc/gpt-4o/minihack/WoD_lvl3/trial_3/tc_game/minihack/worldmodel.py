# make sure to include these import statements
from copy import deepcopy

directions = {
    'left': [-1, 0],
    'right': [1, 0],
    'up': [0, 1],
    'down': [0, -1],
    'up_right': [1, 1],
    'down_right': [1, -1],
    'down_left': [-1, -1],
    'up_left': [-1, 1],
}

def transition_model(state, action):
    # Deep copy the state to avoid modifying the original state
    new_state = deepcopy(state)

    # Retrieve agent and minotaur positions
    agent_pos = new_state.get('agent')[0]
    minotaur_pos = new_state.get('minotaur')[0]
    wand_pos = new_state.get('wand')[0] if new_state.get('wand') else None

    # Handle movement actions
    if action in directions:
        direction = directions[action]
        new_agent_pos = [agent_pos[0] + direction[0], agent_pos[1] + direction[1]]

        # Check if the move is valid (i.e., not into a wall)
        if new_agent_pos not in new_state.get('wall'):
            new_state['agent'] = [new_agent_pos]

    # Handle wand pickup
    elif action == 'pickup':
        if agent_pos == wand_pos:
            new_state['inventory'].append('wand')
            new_state.pop('wand', None)  # Remove wand from ground

    # Shooting mechanisms
    elif action in ['zap', 'select_f', 'shoot_up', 'shoot_right', 'shoot_down', 'shoot_left']:
        # Check if the wand is in inventory
        if 'wand' in new_state.get('inventory', []):
            if action == 'zap':
                # No effect on state directly, but part of shooting sequence
                new_state['zap_ready'] = True
            elif action == 'select_f' and new_state.get('zap_ready'):
                # Again, no immediate state effect, part of shooting sequence
                new_state['select_ready'] = True
            elif action.startswith('shoot_') and new_state.get('zap_ready') and new_state.get('select_ready'):
                shoot_direction = action.split('_')[1]
                direction = directions[shoot_direction]

                # Trace shot path up to 5 squares
                for i in range(1, 6):
                    shot_pos = [agent_pos[0] + i * direction[0], agent_pos[1] + i * direction[1]]
                    if shot_pos == minotaur_pos:
                        new_state['minotaur'] = []  # Minotaur is defeated
                        new_state['won'] = True
                        break  # Stop trace once the minotaur is hit

                # Reset shooting preparation flags
                new_state['zap_ready'] = False
                new_state['select_ready'] = False

    # Return the new state
    return new_state