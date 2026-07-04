def killed(state, attacker, target):
    """
    Returns True if the target object is not present in the state, indicating it has been 'killed' by the attacker.
    
    Parameters:
    - state: dict with keys as object names and values as positions or states
    - attacker: object name (e.g., 'agent')
    - target: object name (e.g., 'minotaur')

    Returns:
    - bool: True if the target's key is not present in the state dictionary (i.e., it has been 'killed'), False otherwise
    """
    target_pos = state.get(target)
    return target_pos == [] or target_pos is None