def descended(state, x, y):
    """
    Returns True if object x has descended on or through object y, based on their positions.

    Parameters:
    - state: dict with keys as object names and values as lists of [x, y] positions
    - x: object name (e.g., 'human_rogue_called_agent')
    - y: object name (e.g., 'staircase_down')

    Returns:
    - bool: True if x is at the same coordinates as y indicating descent, False otherwise
    """
    pos_x = state.get(x)
    pos_y = state.get(y)
    if not pos_x or not pos_y:
        return False
    # Assuming descent means reaching the exact position of the staircase
    return pos_x[0] == pos_y[0]