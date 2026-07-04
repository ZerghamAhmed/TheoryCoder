def descended(state, agent, staircase):
    """
    Returns True if the agent has descended the given staircase,
    i.e., the agent's coordinates match the staircase's coordinates.

    Parameters:
    - state: dict mapping object names to lists of positions (e.g., [[x,y]])
    - agent: string name of the agent object
    - staircase: string name of the staircase object

    Returns:
    - bool: True if agent is at the same position as the staircase, False otherwise
    """
    # get the single position entry for agent and staircase
    agent_pos_list = state.get(agent)
    stair_pos_list = state.get(staircase)

    # if either is missing or empty, the predicate is false
    if not agent_pos_list or not stair_pos_list:
        return False

    # compare the first (and only) coordinate pair
    return agent_pos_list[0] == stair_pos_list[0]