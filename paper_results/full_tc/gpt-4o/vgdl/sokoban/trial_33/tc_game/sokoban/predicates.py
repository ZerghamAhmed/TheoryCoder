def ontop(state, obj1, obj2):
    """
    Returns True if obj1 is on top of obj2 based on their current positions in the state.

    Parameters:
    - state: dict representing the current state of the world
    - obj1: first object name (e.g., 'avatar')
    - obj2: second object name (e.g., 'goal')

    Returns:
    - bool: True if obj1's position equals obj2's position, False otherwise
    """
    pos1 = state.get(obj1, [])
    pos2 = state.get(obj2, [])
    
    # Ensure both objects have a valid position
    if not pos1 or not pos2:
        return False
    
    # Since positions are lists of [x, y], check for overlap
    return pos1[0] == pos2[0]