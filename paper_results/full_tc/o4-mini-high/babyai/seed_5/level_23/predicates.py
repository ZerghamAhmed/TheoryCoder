def has(state, agent, key):
    """
    Returns True if the given agent has picked up the given key.

    Parameters:
    - state: dict expected to contain an "agent_carrying" entry
             mapping to a list of keys currently held by the agent.
    - agent: str, name of an agent (e.g. "red_agent")
    - key:   str, name of a key (e.g. "yellow_key")

    Returns:
    - bool: True if `key` is in state["agent_carrying"], False otherwise.
    """
    carrying = state.get('agent_carrying', [])
    return isinstance(carrying, list) and key in carrying

def open(state, door):
    """
    Returns True if the given door is open in the current state.

    Parameters:
    - state: dict expected to contain an entry for each door, mapping to a dict
             with at least an "open" boolean field.
    - door:  str, name of a door (e.g., "red_door_1")

    Returns:
    - bool: True if state[door]["open"] is True, False otherwise.
    """
    door_info = state.get(door)
    if not isinstance(door_info, dict):
        return False
    return bool(door_info.get('open', False))