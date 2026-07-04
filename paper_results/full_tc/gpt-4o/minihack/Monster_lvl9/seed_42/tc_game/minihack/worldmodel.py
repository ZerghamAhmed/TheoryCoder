# make sure to include these import statements
from minihack_utils import directions

def transition_model(state, action):
    # Get the current position of the 'human_rogue_called_agent'
    agent_pos = state.get('human_rogue_called_agent', [[0, 0]])
    
    # Extract the coordinates of the agent
    x, y = agent_pos[0]
    
    # Calculate the potential new position by applying the action vector
    dx, dy = directions.get(action, [0, 0])
    new_x, new_y = x + dx, y + dy
    
    # Constraints - no movement possible if:
    # 1. Agent is on the staircase_up and tries to move in the blocked state
    if state.get('staircase_up', [[-1, -1]]) == [[new_x, new_y]]:
        # If moving to a staircase_up position, check if the action is legal
        if action in ['up', 'up_right', 'right']:
            new_x, new_y = x, y  # Restore the original position
    
    # Build the new state
    new_state = {
        'human_rogue_called_agent': [[new_x, new_y]],
        'staircase_down': state.get('staircase_down', []),
        'staircase_up': state.get('staircase_up', []),
        'inventory': state.get('inventory', []),
        'won': state.get('won', False),
        'lost': state.get('lost', False)
    }

    # Return the new state after the action
    return new_state