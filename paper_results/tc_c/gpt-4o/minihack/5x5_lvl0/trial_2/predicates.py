def descended(state, x, y):
    """
    Returns True if object x has descended to object y.
    
    Parameters:
    - state: dict representing the current world state
    - x: object name (e.g., 'human_rogue_called_agent')
    - y: object name (e.g., 'staircase_down')

    Returns:
    - bool: True if x is at the same position as y, indicating it has descended, False otherwise
    """
    # Get the current positions of x and y
    pos_x = state.get(x, [[0, 0]])[0]
    pos_y = state.get(y, [[0, 0]])[0]
    
    # Check if the positions match
    return pos_x == pos_y