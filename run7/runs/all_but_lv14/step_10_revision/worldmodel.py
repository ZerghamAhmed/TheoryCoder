# make sure to include these import statements
from copy import deepcopy

# Utility for directions
directions = {
    'left': [-1, 0],
    'right': [1, 0],
    'up': [0, 1],
    'down': [0, -1],
}

def transition_model(state, action):
    # Create a deep copy of the state to modify
    new_state = deepcopy(state)

    # Extract necessary state components
    controllables = state['controllables']
    pushables = state['pushables']
    borders = state['border']
    rules_formed = state['rules_formed']
    
    # Get the direction vector from the action
    dx, dy = directions[action]

    def move_object(obj_pos, dx, dy):
        """ Helper function to calculate new position """
        return [obj_pos[0] + dx, obj_pos[1] + dy]

    def is_within_bounds(pos):
        """ Check if the position is within the borders """
        return pos not in borders

    def is_pushable_at_position(pos):
        """ Check if any pushable object is at a specific position """
        for pushable in pushables:
            if pos in new_state[pushable]:
                return pushable
        return None

    def is_stop_at_position(pos):
        """ Check if there is a stop object at a specific position """
        if 'wall_word is_word stop_word' in rules_formed:
            if pos in new_state.get('wall_obj', []):
                return True
        return False

    def move_controllable(controllable):
        """ Move the controllable object and handle interactions """
        current_pos = state[controllable][0]  # Assuming single position per controllable
        new_pos = move_object(current_pos, dx, dy)

        if not is_within_bounds(new_pos) or is_stop_at_position(new_pos):
            return  # Can't move out of bounds or into a stop object

        # Check for pushable at the new position
        pushable = is_pushable_at_position(new_pos)

        if pushable:
            # Attempt to move the pushable
            new_pushable_pos = move_object(new_pos, dx, dy)
            if is_within_bounds(new_pushable_pos) and not is_pushable_at_position(new_pushable_pos) and not is_stop_at_position(new_pushable_pos):
                # Move the pushable
                pushable_index = new_state[pushable].index(new_pos)
                new_state[pushable][pushable_index] = new_pushable_pos
            else:
                return  # Can't move if pushable can't be moved

        # Check for sink interaction
        if 'goop_word is_word sink_word' in rules_formed:
            if new_pos in new_state.get('goop_obj', []):
                # Remove the controllable and goop_obj at this position
                new_state[controllable] = []
                new_state['controllables'] = []
                new_state['goop_obj'].remove(new_pos)
                new_state['lost'] = True
                return
        
        # Move the controllable object
        new_state[controllable][0] = new_pos

    # Additional logic to handle pushables interacting with sink
    def handle_pushable_sink_interaction():
        """ Handle interactions where a pushable and sink (goop) overlap """
        for pushable in pushables:
            for position in new_state[pushable][:]:
                if 'goop_word is_word sink_word' in rules_formed:
                    if position in new_state.get('goop_obj', []):
                        # Remove both pushable and goop_obj at this position
                        new_state[pushable].remove(position)
                        new_state['goop_obj'].remove(position)
                        # Ensure pushables list is not modified incorrectly
                        if not new_state[pushable]:
                            new_state['pushables'].remove(pushable)

    # Loop through all controllables and move them
    for controllable in controllables:
        move_controllable(controllable)

    # Handle pushable and sink interactions
    handle_pushable_sink_interaction()

    # Handle rule-based transformations
    if 'rock_word is_word flag_word' in rules_formed:
        # Change rock_obj to flag_obj
        if 'rock_obj' in new_state:
            new_state['flag_obj'] = new_state.pop('rock_obj')
        
        # Update overlappables if necessary
        if 'flag_obj' in new_state:
            new_state['overlappables'].append('flag_obj')
    
    return new_state