def won(state):
    """
    Returns True if the 'won' predicate holds in the given state.

    Parameters:
    - state: dict containing the key 'won' with a boolean value

    Returns:
    - bool: True if state['won'] is True, False otherwise
    """
    return state.get('won', False)