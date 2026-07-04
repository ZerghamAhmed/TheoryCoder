def descended(state, x, y):
    """
    Returns True if object x has descended to object y based on their coordinates.

    Parameters:
    - state: dict with keys as object names and values as [x, y] positions
    - x: object name (e.g., 'human_rogue_called_agent')
    - y: object name (e.g., 'staircase_down')

    Returns:
    - bool: True if x's position matches y's position, False otherwise
    """
    pos_x = state.get(x)
    pos_y = state.get(y)
    
    if pos_x is None or pos_y is None:
        return False
    
    # Check if the positions of x and y match
    return pos_x[0] == pos_y[0]