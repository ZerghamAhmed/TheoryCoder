def killed(state, attacker, target):
    """
    Returns True if the target is considered 'killed', which for the purposes of this
    world model means the 'target' key is absent from the state or its value is an empty list.

    Parameters:
    - state: dict that includes keys like 'minotaur', 'agent', etc. with their positions as values
    - attacker: object name (e.g., 'agent')
    - target: object name (e.g., 'minotaur')

    Returns:
    - bool: True if the target is absent from the state dictionary or if its positions list is empty
    """
    target_positions = state.get(target)
    return target_positions is None or len(target_positions) == 0