def reached(state, arg1, arg2):
    """
    Returns True if arg1's position (e.g., 'avatar') matches arg2's position 
    (e.g., 'goal') in the given state.

    Parameters:
    - state: dict containing the positions of 'avatar', 'goal', and other objects.
    - arg1: the name of the first object (e.g., 'avatar').
    - arg2: the name of the second object (e.g., 'goal').

    Returns:
    - bool: True if the position of arg1 matches the position of arg2, False otherwise.
    """
    pos1 = state.get(arg1, [[]])[0]  # Fetch the position of arg1 (e.g., avatar)
    pos2 = state.get(arg2, [[]])[0]  # Fetch the position of arg2 (e.g., goal)

    return pos1 == pos2