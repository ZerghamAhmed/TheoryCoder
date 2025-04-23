from copy import deepcopy

# Directions for turning the agent
directions = {
    'left': [0, -1],
    'right': [0, 1],
    'up': [-1, 0],
    'down': [1, 0],
}

def transition_model(state, action):
    new_state = deepcopy(state)
    
    # Get the agent's current position and direction
    agent_pos = new_state['red_agent'][0]
    agent_dir = new_state['agent_direction']
    agent_carrying = new_state['agent_carrying'][0]
    
    # Calculate the position in front of the agent
    front_pos = [agent_pos[0] + agent_dir[0], agent_pos[1] + agent_dir[1]]
    
    if action == 'left':
        # Rotate the agent left
        new_state['agent_direction'] = [-agent_dir[1], agent_dir[0]]
        
    elif action == 'right':
        # Rotate the agent right
        new_state['agent_direction'] = [agent_dir[1], -agent_dir[0]]
        
    elif action == 'forward':
        # Move the agent forward if not blocked by a wall or object
        if front_pos not in new_state.get('grey_wall', []) and \
           not any(front_pos in positions for key, positions in new_state.items() if key not in ['grey_wall', 'red_agent', 'agent_direction', 'agent_carrying']):
            new_state['red_agent'][0] = front_pos
    
    elif action == 'pickup':
        # Attempt to pick up an item
        if agent_carrying['type'] == 'none':
            for item_type, positions in new_state.items():
                if item_type not in ['grey_wall', 'red_agent', 'agent_direction', 'agent_carrying']:
                    if front_pos in positions:
                        # Update carrying state and remove the item from the map
                        item_color = item_type.split('_')[0]
                        item_kind = item_type.split('_')[1]
                        new_state['agent_carrying'] = [{'type': item_kind, 'color': item_color}]
                        positions.remove(front_pos)
                        break
    
    elif action == 'drop':
        # Drop the item if not blocked and not one square away from a doorway
        if agent_carrying['type'] != 'none':
            item_pos = new_state.get(f"{agent_carrying['color']}_{agent_carrying['type']}", [])
            if front_pos not in item_pos and not any(front_pos in positions for key, positions in new_state.items() if key.startswith('open_') or key.startswith('locked_')):
                # Drop the item in front of the agent
                item_pos.append(front_pos)
                new_state['agent_carrying'] = [{'type': 'none', 'color': 'none'}]
                
    elif action == 'toggle':
        # Toggle action
        for item_type, positions in new_state.items():
            if item_type.endswith('_box') and front_pos in positions:
                color = item_type.split('_')[0]
                # Check for a locked door of the same color
                locked_door_present = any(key.startswith(f'locked_{color}_door') for key in new_state)
                key_exists = any(key.startswith(f'{color}_key') for key in new_state)
                
                if locked_door_present and not key_exists:
                    # Transform the box into a key
                    positions.remove(front_pos)
                    new_state.setdefault(f'{color}_key', []).append(front_pos)
                break
            elif item_type.startswith('closed') and front_pos in positions:
                # Open the door
                open_door = f"open_{item_type.split('_')[1]}"
                positions.remove(front_pos)
                new_state.setdefault(open_door, []).append(front_pos)
                break
            elif item_type.startswith('locked') and front_pos in positions:
                # Check if carrying the correct key
                color = item_type.split('_')[1]
                if agent_carrying['type'] == 'key' and agent_carrying['color'] == color:
                    open_door = f"open_{color}_door"
                    positions.remove(front_pos)
                    new_state.setdefault(open_door, []).append(front_pos)
                break

    return new_state
