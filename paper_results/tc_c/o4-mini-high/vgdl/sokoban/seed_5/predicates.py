def overlaps(state, obj1, obj2):
    """
    Returns True if any position of object `obj1` coincides with any position of object `obj2`.

    Parameters:
    - state: dict mapping object‐type keys to lists of [x, y] positions
    - obj1: key in state (e.g. 'box')
    - obj2: key in state (e.g. 'hole')

    Returns:
    - bool: True if an (x, y) in state[obj1] equals an (x, y) in state[obj2], False otherwise
    """
    positions1 = state.get(obj1, [])
    positions2 = state.get(obj2, [])
    # if either list is empty, they cannot overlap
    if not positions1 or not positions2:
        return False
    # check for any matching coordinate
    for pos in positions1:
        if pos in positions2:
            return True
    return False