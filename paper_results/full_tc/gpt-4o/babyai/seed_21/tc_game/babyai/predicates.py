def carrying(state, agent, key):
    """
    Returns True if the specified agent is carrying the specified key.

    Parameters:
    - state: dict representing the raw state of the grid.
    - agent: the agent's identifier (e.g., "red_agent").
    - key: the key's identifier (e.g., "yellow_key").

    Returns:
    - bool: True if the agent is carrying the key, False otherwise.
    """
    return key in state.get('agent_carrying', [])

def unlocked(state, door):
    """
    Returns True if the specified door is unlocked.

    Parameters:
    - state: dict representing the raw state of the grid.
    - door: the door's identifier (e.g., "red_door_1").

    Returns:
    - bool: True if the door is unlocked, False otherwise.
    """
    door_state = state.get(door, {})
    return not door_state.get('locked', True)