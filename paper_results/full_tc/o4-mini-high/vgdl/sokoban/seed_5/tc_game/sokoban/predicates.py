def overlaps(state, obj1, obj2):
    """
    Returns True if object obj1 occupies the same grid cell as object obj2.

    Parameters:
    - state: dict mapping object‐type keys to lists of [x, y] positions
             e.g. state['avatar'] = [[2,5]], state['goal'] = [[4,1]]
    - obj1: string name of first object (e.g. 'avatar')
    - obj2: string name of second object (e.g. 'goal')

    Returns:
    - bool: True if any position of obj1 matches any position of obj2
    """
    positions1 = state.get(obj1, [])
    positions2 = state.get(obj2, [])
    for pos in positions1:
        if pos in positions2:
            return True
    return False