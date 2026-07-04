def overlaps(state, obj1, obj2):
    """
    Returns True if object obj1 occupies the same grid cell as object obj2.

    Parameters:
    - state: dict with keys 'avatar', 'wall', 'floor', 'trap', 'cheese'
             Each maps to a list of [x, y] coordinate lists.
    - obj1: name of the first object (e.g., 'avatar')
    - obj2: name of the second object (e.g., 'cheese')

    Returns:
    - bool: True if any position of obj1 coincides with any position of obj2.
    """
    positions1 = state.get(obj1, [])
    positions2 = state.get(obj2, [])
    # If either object has no known positions, they cannot overlap.
    if not positions1 or not positions2:
        return False

    # Check all pairs of positions for equality.
    for p1 in positions1:
        for p2 in positions2:
            if p1 == p2:
                return True
    return False