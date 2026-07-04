def carrying(state, agent, key):
    """
    Returns True if the given agent is carrying the given key in the state.

    Parameters:
    - state: dict with keys from the raw state (e.g. "agent_carrying")
    - agent: str, name of an agent (e.g. "red_agent")
    - key: str, name of a key (e.g. "yellow_key")

    Returns:
    - bool: True if `key` is listed in state["agent_carrying"], False otherwise.
    """
    # Ensure the agent exists in the state
    if agent not in state:
        return False

    # Get the list of keys the agent is carrying (default to empty list)
    carried_keys = state.get("agent_carrying", [])
    return key in carried_keys