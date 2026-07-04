def opened(state, d):
    """
    Returns True if the door `d` is opened in the given state.

    Parameters:
    - state (dict): The current world state.
    - d (str): The name of the door object.

    Returns:
    - bool: True if `d` is open, False otherwise.
    """
    door_info = state.get(d)
    if not isinstance(door_info, dict):
        return False
    return bool(door_info.get("open", False))