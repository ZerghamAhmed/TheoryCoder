def killed(state, agent, target):
    """
    Returns True if `agent` has killed `target`, i.e. the target object
    no longer exists in the world-state dictionary.

    Parameters:
    - state: dict representing the current world state
    - agent: string name of the agent (e.g., 'agent')
    - target: string name of the object to check (e.g., 'minotaur')

    Returns:
    - bool: True if `target` is absent or its entry is empty, False otherwise.
    """
    return target not in state or not state.get(target)