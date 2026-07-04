def carrying(state, agent, key):
    """
    Returns True if the given agent is carrying the given key in the current state.

    Parameters:
    - state: dict containing the world state. Must include the key "agent_carrying"
             which is a list of keys currently held by the (single) agent.
    - agent: str name of the agent (e.g. "red_agent")
    - key:   str name of the key (e.g. "yellow_key")

    Returns:
    - bool: True if `key` is in state["agent_carrying"] and `agent` exists in state, False otherwise.
    """
    if agent not in state:
        return False
    carried_keys = state.get('agent_carrying', [])
    return key in carried_keys

def opened(state, door):
    """
    Returns True if the given door is open in the current state.

    Parameters:
    - state: dict containing the world state. Doors are represented as keys
             whose values are dicts with an "open" boolean field.
    - door:  str name of the door (e.g., "red_door_1")

    Returns:
    - bool: True if `door` exists in state and state[door]["open"] is True,
            False otherwise.
    """
    if door not in state:
        return False
    door_info = state.get(door)
    return bool(door_info.get('open', False))