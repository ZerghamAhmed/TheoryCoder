def opened(state, d):
    """
    Returns True if door `d` is opened in the given state.

    Parameters:
    - state: dict mapping object names to their properties (raw state).
    - d: the name of the door (string).

    Returns:
    - bool: True if `d` exists in state and its 'open' attribute is True, False otherwise.
    """
    door = state.get(d)
    if not isinstance(door, dict):
        return False
    return door.get("open", False)