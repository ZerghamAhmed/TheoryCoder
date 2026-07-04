def descended(state, agent, staircase):
    """
    Returns True if the agent has descended the given staircase,
    i.e., if the agent's current position matches the staircase's position.

    Parameters:
    - state: dict mapping object names to lists of [x, y] coordinates
    - agent: object name for the agent (e.g., 'human_rogue_called_agent')
    - staircase: object name for the staircase (e.g., 'staircase_down')

    Returns:
    - bool: True if the agent and staircase occupy the same cell, False otherwise
    """
    agent_positions = state.get(agent, [])
    staircase_positions = state.get(staircase, [])
    if not agent_positions or not staircase_positions:
        return False

    # each entry is a [x, y] pair; we assume one instance per object
    return agent_positions[0] == staircase_positions[0]