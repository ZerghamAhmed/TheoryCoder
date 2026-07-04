def overlap(state, x, y):
    """
    Returns True if object x overlaps with object y (i.e., they occupy the same coordinates).

    Parameters:
    - state: dict mapping object names to [[x, y]] positions
    - x: object name (e.g., 'human_rogue_called_agent')
    - y: object name (e.g., 'staircase_down')

    Returns:
    - bool: True if x and y have the same [x, y] coordinates in the state, False otherwise
    """
    pos_list_x = state.get(x)
    pos_list_y = state.get(y)

    # Both objects must exist and have at least one coordinate entry
    if not pos_list_x or not pos_list_y:
        return False

    # Extract their first (and only) positions
    coord_x = pos_list_x[0]
    coord_y = pos_list_y[0]

    # They overlap if and only if both coordinates match
    return coord_x[0] == coord_y[0] and coord_x[1] == coord_y[1]