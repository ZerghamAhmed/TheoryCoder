def reaches(state, x, y):
    """
    Returns True if object x has reached object y, i.e. they occupy the same grid cell.

    Parameters:
    - state: dict mapping entity names to lists of [x,y] positions
    - x: name of the first object (e.g. 'avatar')
    - y: name of the second object (e.g. 'goal')

    Returns:
    - bool: True if any position of x matches any position of y, False otherwise
    """
    positions_x = state.get(x, [])
    positions_y = state.get(y, [])
    for pos_x in positions_x:
        if pos_x in positions_y:
            return True
    return False