def overlap(state, arg1, arg2):
    """
    Returns True if arg1 and arg2 overlap (i.e., occupy the same coordinates).

    Parameters:
    - state: dict mapping object names to lists of [x, y] positions (or other keys)
    - arg1: object name (str)
    - arg2: object name (str)

    Returns:
    - bool: True if arg1 and arg2 share the same [x, y] coordinates, False otherwise
    """
    pos_list1 = state.get(arg1)
    pos_list2 = state.get(arg2)
    # Both objects must exist and have at least one coordinate entry
    if not pos_list1 or not pos_list2:
        return False
    # Compare their first (and only) positions
    return pos_list1[0] == pos_list2[0]


def won(state):
    """
    Returns True if the 'won' predicate is satisfied in the current state.

    Parameters:
    - state: dict containing a boolean under the key 'won'

    Returns:
    - bool: True if state['won'] is True, False otherwise
    """
    return bool(state.get('won', False))