def isOpen(state, d):
    """
    Returns True if door d is open in the given state.

    Parameters:
    - state: dict containing the raw state (e.g., keys like 'red_door_1' mapping to dicts with attributes).
    - d:    str, the name of a door object.

    Returns:
    - bool: True if state[d]["open"] is True, False otherwise.
    """
    door_info = state.get(d)
    if door_info is None:
        return False
    return door_info.get("open", False)