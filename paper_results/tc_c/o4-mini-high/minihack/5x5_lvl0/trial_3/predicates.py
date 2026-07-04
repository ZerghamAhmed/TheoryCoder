def overlap(state, obj1, obj2):
    """
    Returns True if obj1 overlaps obj2 based on their [x, y] coordinates in the state.

    Parameters:
    - state: dict mapping object names to lists of [x, y] positions
    - obj1: name of the first object (string)
    - obj2: name of the second object (string)

    Returns:
    - bool: True if obj1 and obj2 occupy the same coordinates, False otherwise
    """
    pos_list1 = state.get(obj1)
    pos_list2 = state.get(obj2)
    if not pos_list1 or not pos_list2:
        return False

    x1, y1 = pos_list1[0]
    x2, y2 = pos_list2[0]
    return (x1 == x2) and (y1 == y2)