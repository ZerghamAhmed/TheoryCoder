def reached(state, a, g):
    """
    Returns True if the agent (a) has reached the goal (g) based on their positions.

    Parameters:
    - state: dict with keys as object names and values as position lists.
    - a: agent name (e.g., 'avatar').
    - g: goal name (e.g., 'goal').

    Returns:
    - bool: True if the position of the agent is the same as the position of the goal, False otherwise.
    """
    # Get the positions of the agent and the goal
    agent_pos = state.get(a, [[]])[0]  # Example: state['avatar'] -> [[x, y]]
    goal_pos = state.get(g, [[]])[0]   # Example: state['goal'] -> [[x, y]]

    # Check if both agent and goal positions are valid and the same
    return agent_pos == goal_pos