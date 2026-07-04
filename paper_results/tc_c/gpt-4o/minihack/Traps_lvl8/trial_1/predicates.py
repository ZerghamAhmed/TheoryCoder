def descended(state, x, y):
    """
    Returns True if object x has descended object y, typically a staircase.

    Parameters:
    - state: dict with keys as object names and values as properties or coordinates
    - x: object name (e.g., 'human_rogue_called_agent')
    - y: object name (e.g., 'staircase_down')

    Returns:
    - bool: True if x has descended y, False if not
    """
    # Check if object's x-coordinate and y-coordinate are aligned as a sign of descent
    x_pos = state.get(x, [[0, 0]])[0]
    y_pos = state.get(y, [[0, 0]])[0]
    return x_pos[0] == y_pos[0] and x_pos[1] == y_pos[1]