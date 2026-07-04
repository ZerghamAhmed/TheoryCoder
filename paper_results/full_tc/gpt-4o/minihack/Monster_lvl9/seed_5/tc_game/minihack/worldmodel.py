from minihack_utils import directions

def transition_model(state, action):
    # Copy the current state to avoid modifying it directly
    new_state = state.copy()
    
    # Get current position of the agent
    agent_pos = state.get('human_rogue_called_agent', [[]])[0]
    
    # Get movement offsets for the given action
    move_offset = directions.get(action, [0, 0])
    
    # Calculate new position
    new_pos = [agent_pos[0] + move_offset[0], agent_pos[1] + move_offset[1]]
    
    # Example of determining if a move is blocked (not exhaustive implementation)
    # This assumes that if new_pos is out of bounds or has an obstacle, action is blocked
    # Here, these are not defined, so the code is conceptual
    is_blocked = False  # Add logic to determine if the new position is blocked
    
    # Update position if not blocked
    if not is_blocked:
        new_state['human_rogue_called_agent'] = [new_pos]
    
    return new_state