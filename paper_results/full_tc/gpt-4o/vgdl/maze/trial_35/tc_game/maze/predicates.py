def reached(state, agent, target):
    """
    Returns True if the agent's position matches the target's position.

    Parameters:
    - state: dict with keys as object types and values as corresponding positions
    - agent: object's name (string) whose position is to be checked (e.g., 'avatar')
    - target: object's name (string) representing the goal (e.g., 'goal')

    Returns:
    - bool: True if the agent has reached the target, False otherwise
    """
    agent_pos = state.get(agent, [])
    target_pos = state.get(target, [])

    # Ensure we have valid positions for both agent and target
    if not agent_pos or not target_pos:
        return False
    
    # Compare the positions of the agent and the target
    return agent_pos[0] == target_pos[0]