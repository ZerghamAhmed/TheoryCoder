def overlap(state, obj1, obj2):
    """
    Returns True if obj1 and obj2 occupy the same coordinates in the state.

    Parameters:
    - state: dict mapping object names to position lists, e.g.
             {"human_rogue_called_agent": [[x, y]], "staircase_down": [[x, y]], ...}
    - obj1: name of the first object
    - obj2: name of the second object

    Returns:
    - bool: True if obj1 and obj2 have identical [x, y] coordinates, False otherwise
    """
    pos1_list = state.get(obj1)
    pos2_list = state.get(obj2)

    # If either object is missing or has no position recorded, they cannot overlap
    if not pos1_list or not pos2_list:
        return False

    # Extract the single position for each object
    pos1 = pos1_list[0]
    pos2 = pos2_list[0]

    # Overlap if and only if x and y match exactly
    return pos1[0] == pos2[0] and pos1[1] == pos2[1]