def touching(state, obj1, obj2):
    """
    Returns True if obj1 is touching obj2, i.e. they occupy the same grid cell.

    Parameters:
    - state: dict mapping object names to lists of [x, y] positions
    - obj1: name of the first object (e.g., 'avatar')
    - obj2: name of the second object (e.g., 'cheese')

    Returns:
    - bool: True if any position of obj1 equals any position of obj2, False otherwise
    """
    positions1 = state.get(obj1, [])
    positions2 = state.get(obj2, [])
    for pos1 in positions1:
        for pos2 in positions2:
            if pos1 == pos2:
                return True
    return False