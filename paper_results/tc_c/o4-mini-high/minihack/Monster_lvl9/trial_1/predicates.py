def descended(state, agent, stair):
    """
    Returns True if the agent has descended onto the given stair,
    i.e., the agent's coordinates exactly match the stair's coordinates.

    Parameters:
    - state: dict mapping object names to lists of [x, y] positions
    - agent: name of the agent object (e.g., 'human_rogue_called_agent')
    - stair: name of the stair object (e.g., 'staircase_down')

    Returns:
    - bool: True if agent and stair occupy the same cell, False otherwise
    """
    # fetch the list of positions for agent and stair
    agent_pos_list = state.get(agent)
    stair_pos_list = state.get(stair)

    # if either object is missing or has no position, cannot have descended
    if not agent_pos_list or not stair_pos_list:
        return False

    # extract the single [x, y] coordinate for each
    ax, ay = agent_pos_list[0]
    sx, sy = stair_pos_list[0]

    # descended if agent is exactly on the stair cell
    return (ax == sx) and (ay == sy)