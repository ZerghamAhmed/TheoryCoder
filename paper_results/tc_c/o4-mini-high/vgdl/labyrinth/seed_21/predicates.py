def overlaps(state, x, y):
    """
    Returns True if object x overlaps object y (i.e., they occupy at least one identical grid cell).

    Parameters:
    - state: dict mapping object names to a list of their occupied [x, y] coordinates
    - x: name of the first object (e.g., 'avatar')
    - y: name of the second object (e.g., 'goal')

    Returns:
    - bool: True if any coordinate in state[x] is also in state[y], False otherwise
    """
    coords_x = state.get(x, [])
    coords_y = state.get(y, [])
    for coord in coords_x:
        if coord in coords_y:
            return True
    return False