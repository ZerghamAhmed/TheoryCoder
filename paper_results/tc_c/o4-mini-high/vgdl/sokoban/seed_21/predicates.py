def overlaps(state, x, y):
    """
    Returns True if any position of object x coincides with any position of object y.

    Parameters:
    - state: dict mapping keys (e.g., 'box', 'hole', 'avatar') to lists of [x, y] coordinates
    - x: name of the first object/type (e.g., 'box')
    - y: name of the second object/type (e.g., 'hole')

    Returns:
    - bool: True if there is at least one coordinate pair that appears in both state[x] and state[y]
    """
    coords_x = state.get(x, [])
    coords_y = state.get(y, [])
    # If either list is empty, they cannot overlap
    if not coords_x or not coords_y:
        return False
    # Check for any matching coordinate
    for c in coords_x:
        if c in coords_y:
            return True
    return False