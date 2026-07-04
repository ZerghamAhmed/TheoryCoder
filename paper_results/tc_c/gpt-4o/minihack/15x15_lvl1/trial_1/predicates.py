def descended(state, agent, staircase):
    """
    Returns True if the agent has descended the staircase, based on their proximity.

    Parameters:
    - state: dict with keys as object names and values as [x, y] positions
    - agent: object name representing the agent (e.g., 'human_rogue_called_agent')
    - staircase: object name representing the staircase (e.g., 'staircase_down')

    Returns:
    - bool: True if the agent's position is at the staircase's position, False otherwise
    """
    agent_pos = state.get(agent)
    staircase_pos = state.get(staircase)
    if agent_pos is None or staircase_pos is None:
        return False
    return agent_pos[0] == staircase_pos[0] and agent_pos[1] == staircase_pos[1
        ]