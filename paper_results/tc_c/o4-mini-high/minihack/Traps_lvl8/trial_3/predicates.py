def descended(state, arg1, arg2):
    """
    Returns True if object arg1 has 'descended' to object arg2,
    which in this low‐level model means they occupy the same coordinates.

    Parameters:
    - state: dict mapping object names to lists of [x, y] positions
    - arg1: name of the first object (e.g., 'human_rogue_called_agent')
    - arg2: name of the second object (e.g., 'staircase_down')

    Returns:
    - bool: True if arg1's position equals arg2's position, False otherwise
    """
    pos_list1 = state.get(arg1)
    pos_list2 = state.get(arg2)

    # Both objects must have at least one position entry
    if not pos_list1 or not pos_list2:
        return False

    # Compare their first (and in this domain, only) coordinates
    return pos_list1[0] == pos_list2[0]