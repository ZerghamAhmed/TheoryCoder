def isOpen(state, door_name):
    """
    Returns True if the door `door_name` is open in the given state.

    Parameters:
    - state: dict representing the world state. Door entries are expected to be
             keyed by door name and have an "open" boolean field.
    - door_name: string name of the door (e.g., 'yellow_door_1')

    Returns:
    - bool: True if the door exists in state and its "open" field is True;
            False otherwise.
    """
    door = state.get(door_name)
    if door is None:
        return False
    return bool(door.get("open", False))