def ontop(state, arg1, arg2):
    """
    Returns True if the object represented by arg1 is on top of the object represented by arg2.
    
    Parameters:
    - state: dict with keys as object names and values as lists of [x, y] positions
    - arg1: object name (e.g., 'avatar')
    - arg2: object name (e.g., 'goal')
    
    Returns:
    - bool: True if arg1 is on top of arg2, False otherwise
    """
    pos1 = state.get(arg1, [])
    pos2 = state.get(arg2, [])
    return any(p1 == p2 for p1 in pos1 for p2 in pos2)
