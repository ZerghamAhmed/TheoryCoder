def killed(state, attacker, target):
    """
    Returns True if `attacker` has killed `target`, i.e., if the target
    no longer exists in the game state.

    Parameters:
    - state: dict representing the current world state. Keys are object names
      (like 'agent', 'minotaur', etc.) or other world elements ('wall',
      'inventory', 'won', 'lost', ...). For objects, the value is a list
      of [x, y] positions where they exist.
    - attacker: name of the object that would do the killing (e.g., 'agent')
    - target:   name of the object that might be killed (e.g., 'minotaur')

    Returns:
    - bool: True if `target` is absent or has an empty position list in the
      state, indicating it has been killed; False otherwise.
    """
    positions = state.get(target, None)
    if positions is None:
        return True
    if isinstance(positions, list) and len(positions) == 0:
        return True
    return False