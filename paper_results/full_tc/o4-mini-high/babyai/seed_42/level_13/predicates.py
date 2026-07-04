def pickedup(state, agent, key):
    """
    Returns True if the agent `agent` has picked up the key `key`.

    Parameters:
    - state: dict representing the current world state. Must contain the key
             "agent_carrying", whose value is a list of keys the agent is carrying.
    - agent: name of the agent (e.g., 'red_agent')
    - key:   name of the key (e.g., 'yellow_key')

    Returns:
    - bool: True if `key` is in state["agent_carrying"], False otherwise.
    """
    carrying = state.get('agent_carrying', [])
    return key in carrying

def unlocked(state, door):
    """
    Returns True if the door `door` is unlocked.

    Parameters:
    - state: dict representing the current world state. Must contain an entry
             for each door, where the value is a dict with a "locked" boolean.
    - door:  name of the door (e.g., 'red_door_1')

    Returns:
    - bool: True if the door's "locked" field is False, False otherwise.
    """
    door_info = state.get(door)
    if door_info is None:
        return False
    return not door_info.get('locked', True)