def descended(state, obj1, obj2):
    """
    Returns True if obj1 has descended to obj2's location, i.e., if their coordinates match.

    Parameters:
    - state: dict with keys as object names and values as lists of [x, y] positions
    - obj1: object name (e.g., 'human_rogue_called_agent')
    - obj2: object name (e.g., 'staircase_down')

    Returns:
    - bool: True if obj1's position matches obj2's position, False otherwise
    """
    pos1 = state.get(obj1, [[-1, -1]])[0]  # Default to a position that will never match
    pos2 = state.get(obj2, [[-1, -1]])[0]  # Default to a position that will never match
    return pos1 == pos2