def killed(state, attacker, target):
    """
    Returns True if `target` has been killed by `attacker`.
    We infer 'killed' by checking that the target object no longer
    appears in the world-state dictionary.

    Parameters:
    - state: dict representing the game world, mapping object names
             to their positions or other info.
    - attacker: name of the object that would have performed the kill.
    - target:   name of the object being tested for death.

    Returns:
    - bool: True if `target` is absent from `state`, False otherwise.
    """
    # A killed object (like 'minotaur') is removed from the state dict.
    return target not in state