def killed(state, attacker, target):
    """
    Returns True if the target has been killed by the attacker, which means the target's key
    is absent or its position list is empty in the state dict.

    Parameters:
    - state: dict with keys as object names and values as [x, y] positions
    - attacker: object name (e.g., 'agent')
    - target: object name (e.g., 'minotaur')

    Returns:
    - bool: True if the target's position list is empty, indicating it has been killed.
    """
    return not state.get(target)