def descended(state, x, y):
    """
    Returns True if object x has descended object y by checking if their
    coordinates match. This assumes the object's position will change to y's
    position when descending.

    Parameters:
    - state: dict with keys as object names and values as [x, y] positions.
    - x: object name that is supposed to descend (e.g., 'human_rogue_called_agent').
    - y: object name being descended to (e.g., 'staircase_down').

    Returns:
    - bool: True if x's current position matches y's, False otherwise.
    """
    pos_x = state.get(x)
    pos_y = state.get(y)
    if pos_x is None or pos_y is None:
        return False
    return pos_x[0] == pos_y[0]