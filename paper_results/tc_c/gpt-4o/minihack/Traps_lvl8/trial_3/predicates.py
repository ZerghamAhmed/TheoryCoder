def ontop(state, x, y):
    """
    Returns True if the object x is on top of object y based on their y-coordinates.

    Parameters:
    - state: dict with keys as object names and values as lists of [x, y] positions
    - x: object name (e.g., 'human_rogue_called_agent')
    - y: object name (e.g., 'staircase_down')

    Returns:
    - bool: True if x's position is the same as y's position, False otherwise
    """
    pos_x = state.get(x)
    pos_y = state.get(y)
    if pos_x is None or pos_y is None:
        return False
    # Check if object x and object y have the same position
    return pos_x[0] == pos_y[0]