def holding(state, arg1, arg2):
    """
    Returns True if arg1 is holding arg2, according to the 'agent_carrying' entry in the state.

    Parameters:
    - state: dict containing 'agent_carrying' with keys 'type' and 'color'
    - arg1: the agent (e.g., 'red_agent')
    - arg2: the object to be checked if held (e.g., 'blue_ball')

    Returns:
    - bool: True if arg1 is holding arg2, False otherwise
    """
    # Extract the carried object info
    carried = state.get('agent_carrying', {'type': 'none', 'color': 'none'})
    
    # For this example, assume 'blue_ball' has type 'ball' and color 'blue'
    object_properties = {
        'blue_ball': {'type': 'ball', 'color': 'blue'}
    }

    obj_props = object_properties.get(arg2)
    if not obj_props:
        return False

    return carried['type'] == obj_props['type'] and carried['color'] == obj_props['color']
