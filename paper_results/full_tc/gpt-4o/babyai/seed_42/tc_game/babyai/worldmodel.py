# make sure to include these import statements
from copy import deepcopy
from utils import directions

def transition_model(state, action):
    # Clone the state deeply to avoid mutating the original
    new_state = deepcopy(state)
    
    # Extract relevant agent info
    agent_position = new_state.get('red_agent', [[None, None]])[0]
    agent_direction = new_state.get('agent_direction', [0, -1])
    carrying = new_state.get('agent_carrying', [])
    
    # Calculate the position the agent is facing
    facing_x = agent_position[0] + agent_direction[0]
    facing_y = agent_position[1] + agent_direction[1]
    facing_pos = [facing_x, facing_y]
    
    # Helper function to determine if a position is blocked
    def is_blocked(position):
        # Walls and closed doors (that are not open) block movement
        if position in new_state.get('grey_wall', []):
            return True
        for obj, details in new_state.items():
            if not isinstance(details, list) and 'door' in obj:
                if details.get('location') == position and not details.get('open', False):
                    return True
            elif isinstance(details, list):
                if position in details and obj != 'red_agent':
                    return True
        return False
    
    # Action handling
    if action == 'forward':
        # Calculate the new position
        new_x, new_y = agent_position[0] + agent_direction[0], agent_position[1] + agent_direction[1]
        new_position = [new_x, new_y]
        
        # Ensure the move isn't blocked
        if not is_blocked(new_position):
            new_state['red_agent'] = [new_position]
    
    elif action == 'left':
        # Turn left: Rotate direction counter-clockwise
        if agent_direction == [1, 0]:  # Right
            new_state['agent_direction'] = [0, 1]  # Up
        elif agent_direction == [0, 1]:  # Up
            new_state['agent_direction'] = [-1, 0]  # Left
        elif agent_direction == [-1, 0]:  # Left
            new_state['agent_direction'] = [0, -1]  # Down
        elif agent_direction == [0, -1]:  # Down
            new_state['agent_direction'] = [1, 0]  # Right
    
    elif action == 'right':
        # Turn right: Rotate direction clockwise
        if agent_direction == [1, 0]:  # Right
            new_state['agent_direction'] = [0, -1]  # Down
        elif agent_direction == [0, -1]:  # Down
            new_state['agent_direction'] = [-1, 0]  # Left
        elif agent_direction == [-1, 0]:  # Left
            new_state['agent_direction'] = [0, 1]  # Up
        elif agent_direction == [0, 1]:  # Up
            new_state['agent_direction'] = [1, 0]  # Right
    
    elif action == 'pickup':
        # Attempt to pick up an object in the facing position
        for obj, details in list(new_state.items()):
            if obj not in ['agent_direction', 'red_agent', 'agent_carrying', 'grey_wall'] and isinstance(details, list):
                if facing_pos in details:
                    if not carrying:  # Only pick up if not already carrying an item
                        carrying.append(obj)
                        new_state['agent_carrying'] = carrying
                        details.remove(facing_pos)
                    break
    
    elif action == 'drop':
        # Attempt to drop the carried item to the facing position
        if carrying and not is_blocked(facing_pos):
            item_to_drop = carrying.pop()  # Remove item from carrying
            if item_to_drop in new_state:
                new_state[item_to_drop].append(facing_pos)
            else:
                new_state[item_to_drop] = [facing_pos]
            new_state['agent_carrying'] = carrying
    
    elif action == 'toggle':
        # Attempt to toggle a door at the facing position
        for obj, details in new_state.items():
            if 'door' in obj and isinstance(details, dict) and details.get('location') == facing_pos:
                if details.get('locked', False):
                    # If the door is locked, check if carrying the matching key
                    key_color = obj.split('_')[0] + '_key'
                    if key_color in carrying:
                        carrying.remove(key_color)  # Use the key
                        details['locked'] = False
                        details['open'] = True
                elif not details.get('locked', False):
                    # Simply toggle the door open/close state
                    details['open'] = not details.get('open', False)
                break

    return new_state