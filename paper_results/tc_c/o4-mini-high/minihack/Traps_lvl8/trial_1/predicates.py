def won(state, x):
    """
    Returns True if object x is on the down‐staircase location (i.e., has "won" the level).
    
    Parameters:
    - state: dict with keys including
        - '<object_name>': [[x, y]] positions
        - 'staircase_down': [[xd, yd]]
    - x: object name (e.g., 'human_rogue_called_agent')
    
    Returns:
    - bool: True if x's position equals the down‐staircase position, False otherwise
    """
    # get x's position and the down‐staircase position (each stored as [[x, y]])
    pos_x = state.get(x, [[None, None]])[0]
    pos_down = state.get('staircase_down', [[None, None]])[0]
    
    # if either is missing, cannot be on the staircase
    if pos_x[0] is None or pos_down[0] is None:
        return False
    
    # won when exactly on the down‐staircase coords
    return pos_x == pos_down