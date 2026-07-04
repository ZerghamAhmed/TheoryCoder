def descended(state, arg1, arg2):
    """
    Check if the agent has descended a specific staircase.

    Parameters:
    - state: dict representing the current state, includes coordinates of objects
    - arg1: agent name (e.g., 'human_rogue_called_agent')
    - arg2: staircase name (e.g., 'staircase_down')

    Returns:
    - bool: True if agent's position matches the staircase position, False otherwise
    """
    agent_position = state.get(arg1)
    staircase_position = state.get(arg2)

    if agent_position is None or staircase_position is None:
        return False

    # We assume that descending means matching coordinates with the staircase
    return agent_position[0] == staircase_position[0]