def killed(state, agent, minotaur):
    """
    Returns True if `agent` has killed `minotaur`, i.e. the minotaur is no longer present
    in the game‐world state.

    Parameters:
    - state: dict representing the world. Keys include object names mapping to their positions
             (a list of [x,y] coords) or [] if removed.
    - agent: string name of the agent object (unused here but kept for signature consistency)
    - minotaur: string name of the minotaur object

    Returns:
    - bool: True if the minotaur is absent from the map (killed), False otherwise.
    """
    # Look up the minotaur's position list in the state.
    positions = state.get(minotaur)
    # If there's no entry for the minotaur, or its position list is empty, it's killed.
    if positions is None:
        return True
    return isinstance(positions, list) and len(positions) == 0