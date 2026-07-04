def overlap(state, obj1, obj2):
    """
    Returns True if obj1 is overlapping obj2, defined as Chebyshev distance
    between their positions being at most 1.

    Parameters:
    - state: dict with keys as object names and values as lists of one [x, y] position
    - obj1: object name (e.g., 'human_rogue_called_agent')
    - obj2: object name (e.g., 'staircase_up')

    Returns:
    - bool: True if obj1 and obj2 overlap, False otherwise
    """
    pos_list1 = state.get(obj1)
    pos_list2 = state.get(obj2)
    if not pos_list1 or not pos_list2:
        return False
    x1, y1 = pos_list1[0]
    x2, y2 = pos_list2[0]
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    # Chebyshev distance <= 1 means same or adjacent (including diagonals)
    return max(dx, dy) <= 1


def won(state, obj):
    """
    Returns True if the win condition is satisfied for obj.

    Parameters:
    - state: dict containing at least the key 'won' (bool)
    - obj: object name (unused in this simple boolean check)

    Returns:
    - bool: True if state['won'] is True, False otherwise
    """
    return bool(state.get('won', False))