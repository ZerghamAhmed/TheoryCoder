def overlap(state, arg1, arg2):
    """
    Returns True if arg1 overlaps with arg2, i.e., they occupy the same grid coordinates.

    Parameters:
    - state: dict mapping object names to lists of [x, y] coordinates, e.g.
             { 'human_rogue_called_agent': [[32, 13]], 'staircase_down': [[46, 3]], ... }
    - arg1: name of the first object (string)
    - arg2: name of the second object (string)

    Returns:
    - bool: True if both objects exist in state and their coordinates match; False otherwise.
    """
    # Retrieve the list of positions for each object
    pos_list_1 = state.get(arg1)
    pos_list_2 = state.get(arg2)

    # If either object is missing or has no recorded position, they cannot overlap
    if not pos_list_1 or not pos_list_2:
        return False

    # Extract the first (and only) coordinate pair for each
    coord1 = pos_list_1[0]
    coord2 = pos_list_2[0]

    # Check for equality in both x and y
    return coord1[0] == coord2[0] and coord1[1] == coord2[1]