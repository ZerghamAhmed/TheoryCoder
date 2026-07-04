def reaches(state, obj1, obj2):
    """
    Returns True if object obj1 is at the same coordinates as object obj2.

    Parameters:
    - state: dict mapping entity names to lists of [x, y] positions.
             e.g. state['avatar'] == [[1,4]], state['goal'] == [[4,1]]
    - obj1: name of the first object (string), e.g. 'avatar'
    - obj2: name of the second object (string), e.g. 'goal'

    Returns:
    - bool: True if obj1 and obj2 share the same single position, False otherwise.
    """
    # Fetch the lists of positions (default empty if missing)
    pos_list_1 = state.get(obj1, [])
    pos_list_2 = state.get(obj2, [])
    # If either object has no recorded position, cannot have "reached"
    if not pos_list_1 or not pos_list_2:
        return False
    # We assume a single position per object, so compare the first entries
    return pos_list_1[0] == pos_list_2[0]