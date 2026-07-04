def descended(state, x, y):
    """
    Checks if the object `x` has descended to the object `y`.

    Parameters:
    - state: dict with keys as object names and values as lists of [x, y] positions
    - x: object name (e.g., 'human_rogue_called_agent')
    - y: object name (e.g., 'staircase_down')

    Returns:
    - bool: True if the positions of x and y match and represent 'descended' state
    """
    pos_x = state.get(x)
    pos_y = state.get(y)
    if pos_x is None or pos_y is None:
        return False
    return pos_x[0] == pos_y[0] and pos_x[1] == pos_y[1]