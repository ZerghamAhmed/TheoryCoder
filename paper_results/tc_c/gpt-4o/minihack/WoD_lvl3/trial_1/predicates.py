def killed(state, attacker, target):
    """
    Returns True if the target is considered 'killed' based on its absence in the state.

    Parameters:
    - state: dict representing the current game state
    - attacker: name of the attacker (e.g., 'agent')
    - target: name of the target (e.g., 'minotaur')

    Returns:
    - bool: True if the target's position is absent in the state, indicating it is 'killed'
    """
    return target not in state or not state[target]