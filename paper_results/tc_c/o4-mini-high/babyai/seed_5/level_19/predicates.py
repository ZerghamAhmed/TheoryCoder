def carrying(state, agent, key):
    """
    Returns True if the specified agent is carrying the specified key.

    In our state representation, state["agent_carrying"] is a list of keys
    currently held by the (single) agent.  If the given key appears in that
    list, then carrying(agent, key) holds.

    Parameters:
    - state: dict with keys including "agent_carrying"
    - agent: name of the agent (e.g. "red_agent")
    - key: name of the key (e.g. "yellow_key")

    Returns:
    - bool: True if `key` is in state["agent_carrying"], False otherwise
    """
    carried_keys = state.get("agent_carrying", [])
    return key in carried_keys