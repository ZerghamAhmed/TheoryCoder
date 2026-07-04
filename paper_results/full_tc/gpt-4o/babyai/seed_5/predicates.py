def carrying(state, a, k):
    """
    Checks whether the agent is carrying the specified key.
    
    Parameters:
    - state (dict): The current state of the environment, including the keys the agent is carrying.
    - a (str): The agent identifier (e.g., 'red_agent').
    - k (str): The key identifier (e.g., 'yellow_key').

    Returns:
    - bool: True if the agent is currently carrying the specified key, False otherwise.
    """
    return k in state.get('agent_carrying', [])

def door_unlocked(state, d):
    """
    Checks whether the specified door is unlocked.

    Parameters:
    - state (dict): The current state of the environment, including door status.
    - d (str): The door identifier (e.g., 'red_door_1').

    Returns:
    - bool: True if the door is unlocked, False otherwise.
    """
    door_info = state.get(d, {})
    return door_info.get('locked', True) == False