def colocated(state, x, y):
    """
    Returns True if object x and object y share the same grid position.

    Parameters:
    - state: dict mapping object names (like 'avatar', 'goal', etc.) to lists of [x, y] positions.
    - x: name of the first object (string).
    - y: name of the second object (string).

    Returns:
    - bool: True if any position in state[x] matches any position in state[y], False otherwise.
    """
    pos_list_x = state.get(x, [])
    pos_list_y = state.get(y, [])
    if not pos_list_x or not pos_list_y:
        return False

    # Check all combinations of positions
    for px in pos_list_x:
        for py in pos_list_y:
            if px == py:
                return True
    return False