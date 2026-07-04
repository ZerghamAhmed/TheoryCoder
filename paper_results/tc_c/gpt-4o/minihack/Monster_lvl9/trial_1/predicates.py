def descended(state, x, y):
    """
    Returns True if object x has descended to the position of object y.

    Parameters:
    - state: dict with keys as object names and values as lists of [x, y] positions
    - x: object name (e.g., 'human_rogue_called_agent')
    - y: object name (e.g., 'staircase_down')

    Returns:
    - bool: True if the position of x matches the position of y, False otherwise
    """
    pos_x = state.get(x)
    pos_y = state.get(y)
    
    if pos_x is None or pos_y is None:
        return False

    # Assumes that the position is represented as [x, y] and both are lists
    return pos_x[0] == pos_y[0]