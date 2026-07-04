def pickedup(state, ag, k):
    """
    Returns True if agent ag has picked up key k.

    Parameters:
    - state: dict containing the current environment state. Must include the key "agent_carrying",
             which is a list of object names the agent is carrying.
    - ag:    name of the agent (e.g., 'red_agent')
    - k:     name of the key   (e.g., 'yellow_key')

    Returns:
    - bool: True if k is in state["agent_carrying"], False otherwise.
    """
    carrying = state.get("agent_carrying", [])
    return k in carrying