# make sure to include these import statements
from copy import deepcopy
from minihack_utils import directions

def transition_model(state, action):
    # Create a copy of the current state to avoid mutating the input directly
    new_state = deepcopy(state)

    agent_pos = new_state.get('agent')[0]
    wand_pos = new_state.get('wand')[0] if new_state.get('wand') else None
    minotaur_pos = new_state.get('minotaur')[0] if new_state.get('minotaur') else None

    def move_agent(dx, dy):
        """Helper function to move the agent if possible."""
        new_pos = [agent_pos[0] + dx, agent_pos[1] + dy]
        # Check if the new position is not a wall
        if new_pos not in new_state.get('wall', []):
            new_state['agent'] = [new_pos]
            # Wand remains on the ground if the agent is not carrying it
            if wand_pos == new_pos and 'wand' not in new_state['inventory']:
                new_state['wand'] = [new_pos]  # Still on ground as item
            else:
                new_state['wand'] = [] 

    def attempt_pickup():
        """Attempt to pick up the wand if on the same tile."""
        if wand_pos and agent_pos == wand_pos:
            new_state['inventory'].append('wand')
            new_state['wand'] = []

    def attempt_wand_sequence(direction):
        """Attempt to fire the wand in a direction based on proper sequence."""
        sequence_is_ready = 'zap' in new_state['inventory'] and 'f' in new_state['inventory']
        if sequence_is_ready:
            # Remove 'zap' and 'f' from inventory after the sequence is used
            new_state['inventory'] = [item for item in new_state['inventory'] if item not in ['zap', 'f']]
            # Check if the minotaur is within line of fire
            x, y = agent_pos
            dx, dy = directions[direction]
            # Trace the bolt path up to 5 squares
            for _ in range(5):
                x += dx
                y += dy
                if minotaur_pos == [x, y]:  # Successful hit
                    new_state['minotaur'] = []
                    new_state['won'] = True
                    break

    # Process the action effectively using these movements or checks
    if action in directions:
        move_agent(*directions[action])

    elif action == 'pickup':
        attempt_pickup()

    elif action == 'zap':
        if 'wand' in new_state['inventory'] and 'zap' not in new_state['inventory']:
            new_state['inventory'].append('zap')

    elif action == 'select_f':
        if 'zap' in new_state['inventory'] and 'f' not in new_state['inventory']:
            new_state['inventory'].append('f')

    elif action.startswith('shoot_'):
        # Extract the direction part from the action
        direction = action.split('_')[1]
        if direction in directions:
            attempt_wand_sequence(direction)

    return new_state