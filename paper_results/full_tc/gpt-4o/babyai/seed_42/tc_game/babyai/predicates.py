def carrying(state, agent, key):
    """
    Returns True if the agent is carrying the specified key, False otherwise.

    Parameters:
    - state: dict representing the raw state (has an 'agent_carrying' key).
    - agent: the agent's name (e.g., 'red_agent').
    - key: the key's name (e.g., 'yellow_key').

    Returns:
    - bool: True if the agent is carrying the specified key, False otherwise.
    """
    return key in state.get('agent_carrying', [])

def door_unlocked(state, door):
    """
    Returns True if the specified door is unlocked, False otherwise.

    Parameters:
    - state: dict representing the raw state (doors have a "locked" property within their values).
    - door: the door's name (e.g., 'red_door_1').

    Returns:
    - bool: True if the door is unlocked, False otherwise.
    """
    door_info = state.get(door, {})
    return not door_info.get('locked', True)