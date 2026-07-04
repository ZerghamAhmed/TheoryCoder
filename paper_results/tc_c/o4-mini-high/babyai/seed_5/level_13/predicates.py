def opened(state, door):
    """
    Returns True if the specified door is open.

    Parameters:
    - state: dict representing the raw state
    - door: string, the name of a door object (e.g., "purple_door_1")

    Returns:
    - bool: True if state[door]["open"] is True, False otherwise
    """
    door_info = state.get(door)
    if not isinstance(door_info, dict):
        return False
    return door_info.get("open", False)