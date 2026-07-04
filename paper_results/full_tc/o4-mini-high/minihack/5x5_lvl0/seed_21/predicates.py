def descended(state, obj1, obj2):
    """
    Returns True if obj1 has 'descended' onto obj2, i.e., if their coordinates match.

    Parameters:
    - state: dict mapping object names to lists of [x, y] coordinates
    - obj1: name of the first object (e.g., 'human_rogue_called_agent')
    - obj2: name of the second object (e.g., 'staircase_down')

    Returns:
    - bool: True if obj1 is at the exact same [x, y] position as obj2, False otherwise
    """
    pos_list1 = state.get(obj1, [])
    pos_list2 = state.get(obj2, [])
    if not pos_list1 or not pos_list2:
        return False

    # Each entry in state[obj] is a single [x, y] coordinate
    x1, y1 = pos_list1[0]
    x2, y2 = pos_list2[0]
    return (x1 == x2) and (y1 == y2)