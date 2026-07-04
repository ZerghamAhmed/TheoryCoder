def reaches(state, arg1, arg2):
    """
    Returns True if object arg1 reaches object arg2, i.e., they occupy the same position.

    Parameters:
    - state: dict mapping object names to lists of [x, y] positions
             (e.g., 'avatar': [[1, 2]], 'cheese': [[4, 1]])
    - arg1: string name of the first object (e.g., 'avatar')
    - arg2: string name of the second object (e.g., 'cheese')

    Returns:
    - bool: True if both objects have at least one position and their first positions are identical.
    """
    pos_list1 = state.get(arg1, [])
    pos_list2 = state.get(arg2, [])
    if not pos_list1 or not pos_list2:
        return False
    return pos_list1[0] == pos_list2[0]