def reaches(state, obj1, obj2):
    """
    Returns True if the entity named obj1 is at the same position as the
    entity named obj2 in the current state.

    Parameters:
    - state: dict mapping object names to a list of [x, y] positions.
    - obj1: name of the first object (e.g., 'avatar').
    - obj2: name of the second object (e.g., 'goal').

    Returns:
    - bool: True if both objects share the same [x, y] coordinate, False otherwise.
    """
    pos_list_1 = state.get(obj1, [])
    pos_list_2 = state.get(obj2, [])
    if not pos_list_1 or not pos_list_2:
        return False
    # Each list holds a single [x, y] entry for these entities
    return pos_list_1[0] == pos_list_2[0]