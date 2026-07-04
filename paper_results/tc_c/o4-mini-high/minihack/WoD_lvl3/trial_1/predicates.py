def killed(state, a, b):
    """
    Returns True if object b has been killed (i.e., no longer present on the map).

    Parameters:
    - state: dict representing the world (keys are object names, values are lists of positions)
    - a: name of the agent (unused here, but kept for PDDL compatibility)
    - b: name of the object to check (e.g., 'minotaur')

    Returns:
    - bool: True if b is not in the state or its list of positions is empty.
    """
    positions = state.get(b)
    # If b is not even a key in state, or its list is empty, we consider it "killed"
    if positions is None:
        return True
    if isinstance(positions, list) and len(positions) == 0:
        return True
    return False