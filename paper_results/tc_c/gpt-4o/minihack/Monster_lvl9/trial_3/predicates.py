def descended(state, x, y):
    """
    Returns True if object x has descended to object y based on their coordinates

    Parameters:
    - state: dict with keys as object names and values as lists containing position coordinates
    - x: the name of the object attempting to descend (e.g., 'human_rogue_called_agent')
    - y: the name of the object it is trying to descend to (e.g., 'staircase_down')

    Returns:
    - bool: True if x is at the same position as y, indicating it has descended, False otherwise
    """
    pos_x = state.get(x, [[None, None]])[0]
    pos_y = state.get(y, [[None, None]])[0]

    if pos_x is None or pos_y is None:
        return False
    
    return pos_x == pos_y