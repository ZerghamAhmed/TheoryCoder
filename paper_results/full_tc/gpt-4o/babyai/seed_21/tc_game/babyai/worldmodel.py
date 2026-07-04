# make sure to include these import statements
from utils import directions

def transition_model(state, action):
    # Copy state to avoid modifying original
    new_state = state.copy()

    # Extract relevant information
    agent_pos = new_state.get('red_agent', [[]])[0]  # Get agent's position
    agent_dir = new_state.get('agent_direction', [1, 0])  # Get agent's direction
    carrying_items = new_state.get('agent_carrying', [])  # Get carried items

    # Directions map
    direction_map = {
        (1, 0): 'right',
        (-1, 0): 'left',
        (0, 1): 'up',
        (0, -1): 'down',
    }
    
    # Compute the target cell the agent is facing
    target_cell = [agent_pos[0] + agent_dir[0], agent_pos[1] + agent_dir[1]]

    # Handle movement: 'forward'
    if action == 'forward':
        # Check for obstacles such as grey_wall, objects, or closed doors
        for obj_type, obj_positions in new_state.items():
            if isinstance(obj_positions, list):  # Only check list objects
                if target_cell in obj_positions:  # If target cell is blocked
                    return new_state  # No movement, return unchanged state
        
        # If target cell is free, move the agent
        new_state['red_agent'][0] = target_cell

    # Handle rotation: 'left'
    elif action == 'left':
        # Rotate the agent's direction counter-clockwise (left turn)
        if agent_dir == [1, 0]:  # Facing right
            new_state['agent_direction'] = [0, 1]  # Rotate to up
        elif agent_dir == [0, 1]:  # Facing up
            new_state['agent_direction'] = [-1, 0]  # Rotate to left
        elif agent_dir == [-1, 0]:  # Facing left
            new_state['agent_direction'] = [0, -1]  # Rotate to down
        elif agent_dir == [0, -1]:  # Facing down
            new_state['agent_direction'] = [1, 0]  # Rotate to right

    # Handle rotation: 'right'
    elif action == 'right':
        # Rotate the agent's direction clockwise (right turn)
        if agent_dir == [1, 0]:  # Facing right
            new_state['agent_direction'] = [0, -1]  # Rotate to down
        elif agent_dir == [0, -1]:  # Facing down
            new_state['agent_direction'] = [-1, 0]  # Rotate to left
        elif agent_dir == [-1, 0]:  # Facing left
            new_state['agent_direction'] = [0, 1]  # Rotate to up
        elif agent_dir == [0, 1]:  # Facing up
            new_state['agent_direction'] = [1, 0]  # Rotate to right

    # Handle interaction: 'pickup'
    elif action == 'pickup':
        # Look for a pickable object in the target cell (keys, boxes, balls)
        for obj_type, obj_positions in new_state.items():
            if obj_type in ['grey_wall', 'agent_carrying', 'red_agent', 'agent_direction']:
                continue  # Skip walls, agent, and non-pickable items
            if isinstance(obj_positions, list) and target_cell in obj_positions:
                # Pick up the object if the agent is not already carrying something
                if not carrying_items:
                    new_state['agent_carrying'].append(obj_type)  # Add the object to carrying
                    obj_positions.remove(target_cell)  # Remove the object from the world
                break

    # Handle interaction: 'drop'
    elif action == 'drop':
        # Look for a drop target — only if carrying something
        if carrying_items:
            current_item = carrying_items[0]
            # Place the carried object in the target cell
            if target_cell in [pos for positions in new_state.values() if isinstance(positions, list) for pos in positions]:
                pass  # If target cell is blocked, do nothing (invalid action)
            else:
                # Drop the carried object into the world
                if current_item in new_state:
                    new_state[current_item].append(target_cell)
                else:
                    new_state[current_item] = [target_cell]  # Initialize object's location
                new_state['agent_carrying'] = []

    # Handle interaction: 'toggle'
    elif action == 'toggle':
        # Check for a toggleable object in the target cell (e.g., doors)
        for obj_type, obj_data in new_state.items():
            if '_door_' in obj_type and isinstance(obj_data, dict):
                if obj_data['location'] == target_cell:
                    # Toggle if open/close is allowed
                    if obj_data['locked']:
                        # Door is locked; check for a matching key
                        color = obj_type.split('_door_')[0]  # Extract color of the door
                        key = f"{color}_key"
                        if key in carrying_items:
                            obj_data['open'] = not obj_data['open']  # Toggle door open/close
                            obj_data['locked'] = False  # Unlock the door (key used)
                            new_state['agent_carrying'].remove(key)  # Consume the key
                    else:
                        # Door is unlocked; simply toggle open/close
                        obj_data['open'] = not obj_data['open']
                    break

    # Return the updated state
    return new_state