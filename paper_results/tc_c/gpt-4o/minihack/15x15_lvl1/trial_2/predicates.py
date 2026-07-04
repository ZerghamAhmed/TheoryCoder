def descended(state, agent, stairs):
    """
    Returns True if the agent has descended the stairs, based on
    their coordinates.

    Parameters:
    - state: dict with keys as object names and values as [x, y] positions
    - agent: object name for the agent (e.g., 'human_rogue_called_agent')
    - stairs: object name for the stairs (e.g., 'staircase_down')

    Returns:
    - bool: True if agent's coordinates are the same as stairs',
            indicating they've descended, False otherwise
    """
    agent_pos = state.get(agent)
    stairs_pos = state.get(stairs)
    if agent_pos is None or stairs_pos is None:
        return False
    return agent_pos == stairs_pos