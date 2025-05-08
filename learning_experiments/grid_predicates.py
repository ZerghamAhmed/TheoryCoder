def ontop(state, obj1, obj2):
    """
    Returns True if obj1 is on top of obj2, i.e., they share the same coordinates.

    Parameters:
    - state: dict with keys as object names and values as lists of [x, y] positions
    - obj1: object name (e.g., 'avatar')
    - obj2: object name (e.g., 'goal')

    Returns:
    - bool: True if positions are equal, False otherwise
    """
    pos1 = state.get(obj1)
    pos2 = state.get(obj2)
    
    if not pos1 or not pos2:
        return False

    # Assuming each object has only one position, i.e., pos1 and pos2 are lists with one element
    return pos1[0] == pos2[0]
