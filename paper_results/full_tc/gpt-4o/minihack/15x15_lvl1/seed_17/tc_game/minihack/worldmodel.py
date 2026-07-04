# make sure to include these import statements
from minihack_utils import directions

def transition_model(state, action):
    # Extract the current position of the agent
    agent_position = state.get('human_rogue_called_agent', [])[0]

    if not agent_position:
        return state
    
    # Determine the new position based on the action and direction mapping
    move = directions.get(action, [0, 0])
    new_position = [agent_position[0] + move[0], agent_position[1] + move[1]]
    
    # Check if the new position is the same as the starting position when the
    # agent can't move due to an obstacle (like a wall)
    if action in ['up', 'up_left', 'up_right', 'down_left', 'down', 'down_right', 'left', 'right']:
        if new_position == agent_position:
            return state
    
    # Create a new state to return, updating the agent's position
    new_state = state.copy()
    new_state['human_rogue_called_agent'] = [new_position]

    # Return the updated state
    return new_state