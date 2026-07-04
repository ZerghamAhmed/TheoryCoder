def descended(state, agent, stair):
    """
    Checks if the agent is at the same position as the stair, indicating the agent has descended.

    Parameters:
    - state: dict with keys as object names and values as [x, y] positions
    - agent: object name (e.g., 'human_rogue_called_agent')
    - stair: object name (e.g., 'staircase_down')

    Returns:
    - bool: True if the agent is at the stair's position, False otherwise
    """
    agent_pos = state.get(agent, [[-1, -1]])[0]
    stair_pos = state.get(stair, [[-1, -1]])[0]
    
    return agent_pos == stair_pos