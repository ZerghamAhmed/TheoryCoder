# make sure to include these import statements
from copy import deepcopy
from utils import directions

def transition_model(state, action):
    # Create a deep copy of the state to ensure the original state remains intact
    new_state = deepcopy(state)
    
    # Extract agent-related information from the state
    agent_pos = new_state.get('red_agent', [[-1, -1]])[0]  # Agent's position as [x, y]
    agent_direction = new_state.get('agent_direction', [0, 1])  # Agent's facing direction
    agent_carrying = new_state.get('agent_carrying', [])  # Objects the agent is currently carrying

    # Calculate the target cell based on the agent's current direction
    target_cell = [agent_pos[0] + agent_direction[0], agent_pos[1] + agent_direction[1]]

    def is_cell_free(cell):
        """
        Checks if a given cell is free of obstacles like walls, closed doors, or objects.
        """
        for key, attr in new_state.items():
            if key not in ['agent_direction', 'agent_carrying', 'red_agent'] and isinstance(attr, list):
                # Check if cell has any objects (e.g., keys, balls, boxes)
                if cell in attr:
                    return False
            if 'door' in key and isinstance(attr, dict):  
                # Closed doors block movement
                if cell == attr.get('location') and not attr.get('open', False):
                    return False
        return True

    # Handle actions
    if action == 'left':
        # Rotate agent direction counterclockwise (left)
        new_state['agent_direction'] = [-agent_direction[1], agent_direction[0]]
    
    elif action == 'right':
        # Rotate agent direction clockwise (right)
        new_state['agent_direction'] = [agent_direction[1], -agent_direction[0]]
    
    elif action == 'forward':
        # Move agent forward to a free cell (if it's free)
        if is_cell_free(target_cell):
            new_state['red_agent'] = [target_cell]
    
    elif action == 'pickup':
        # Agent picks up a movable object in the target cell (keys, balls, boxes)
        if not agent_carrying:  # Can only carry one item at a time
            for key, attr in new_state.items():
                if isinstance(attr, list) and target_cell in attr:
                    if 'key' in key or 'ball' in key or 'box' in key:  # Identifies movable objects
                        # Add the object to the agent's carrying list and remove it from world
                        new_state['agent_carrying'].append(key)
                        attr.remove(target_cell)
                        break
    
    elif action == 'drop':
        # The agent drops the carried object onto the target cell
        if agent_carrying:
            carried_item = agent_carrying.pop()  # Remove the carried item (only one item allowed)
            if is_cell_free(target_cell):  # Allow drop only on free cells
                new_state[carried_item] = new_state.get(carried_item, []) + [target_cell]
    
    elif action == 'toggle':
        # Used to interact with objects (e.g., open/close doors, unlock locked doors)
        for key, attr in new_state.items():
            if 'door' in key and isinstance(attr, dict):
                if target_cell == attr.get('location'):  # Check if the agent is adjacent to the door
                    if attr.get('locked', False):  # If the door is locked
                        color = key.split('_')[0]  # Extract the door's color from its key
                        if f"{color}_key" in agent_carrying:  # Verify the agent has the right key
                            attr['locked'] = False  # Unlock the door
                            attr['open'] = True  # Unlocking also opens the door
                    elif not attr.get('locked', False):  # If the door is not locked, just toggle open/close
                        attr['open'] = not attr['open']
                    break
    
    # Return the updated state
    return new_state