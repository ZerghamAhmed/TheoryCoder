def overlaps(state, obj1, obj2):
    """
    Returns True if any instance of obj1 occupies the same grid cell as any instance of obj2.

    Parameters:
    - state: dict mapping object‐type keys ('box', 'hole', 'avatar', 'wall', 'floor', etc.)
             to lists of [x, y] coordinates.
    - obj1: str name of the first object type (e.g. 'box')
    - obj2: str name of the second object type (e.g. 'hole')

    Returns:
    - bool: True if there exists at least one coordinate that appears in both
            state[obj1] and state[obj2]; False otherwise.
    """
    coords1 = state.get(obj1, [])
    coords2 = state.get(obj2, [])
    if not coords1 or not coords2:
        return False
    # check if any coordinate is shared
    coords2_set = set(tuple(c) for c in coords2)
    for c in coords1:
        if tuple(c) in coords2_set:
            return True
    return False