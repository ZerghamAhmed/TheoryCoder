def killed(state, attacker, target):
    """
    Returns True if `attacker` has killed `target` in the given state.

    We infer a kill by checking that the `target` no longer appears as an active
    object in the world state (either the key is missing or its position list is empty).

    Parameters:
    - state: dict representing the world (keys like 'agent', 'minotaur', etc.)
    - attacker: string name of the attacker (e.g. 'agent')
    - target: string name of the target (e.g. 'minotaur')

    Returns:
    - bool: True if the target is removed or has an empty position list; False otherwise.
    """
    # Look up the target's entry in the state
    val = state.get(target, None)

    # If there's no entry for the target, or it's an empty list, it's dead/killed.
    if val is None:
        return True
    if isinstance(val, list) and len(val) == 0:
        return True

    # Otherwise, target still exists alive in the world
    return False