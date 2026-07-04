def descended(state, agent, stair):
    """
    Returns True if the agent has descended the given stair, i.e.,
    if the agent's position coincides with the stair's position.

    Parameters:
    - state: dict mapping object names to lists of positions (each a [x, y] list)
    - agent: name of the agent object (e.g., 'human_rogue_called_agent')
    - stair: name of the stair object (e.g., 'staircase_down')

    Returns:
    - bool: True if agent and stair occupy the same coordinate, False otherwise
    """
    agent_positions = state.get(agent)
    stair_positions = state.get(stair)

    # both agent and stair must have exactly one position entry
    if (
        agent_positions is None or stair_positions is None
        or len(agent_positions) != 1 or len(stair_positions) != 1
    ):
        return False

    # compare their single coordinates
    return agent_positions[0] == stair_positions[0]