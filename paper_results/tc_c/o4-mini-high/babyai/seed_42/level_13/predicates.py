def opened(state, d):
    """
    Returns True if door d is marked as opened in the given state.

    Parameters:
    - state: dict containing objects in the world. Doors are represented
             as keys mapping to dicts with an "open" boolean field.
    - d:     str, the name of the door (e.g., "red_door_1").

    Returns:
    - bool: True if the door exists in state and its "open" field is True,
            False otherwise.
    """
    door_info = state.get(d)
    if not isinstance(door_info, dict):
        # Either the door isn't in the state or it's not in the expected format
        return False
    return bool(door_info.get("open", False))