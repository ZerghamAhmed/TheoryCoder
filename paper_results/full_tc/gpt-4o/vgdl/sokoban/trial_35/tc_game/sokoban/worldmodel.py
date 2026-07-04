# make sure to include these import statements
from copy import deepcopy
from utils import directions

def transition_model(state, action):
    # Start by making a deep copy of the state to avoid modifying the original
    new_state = deepcopy(state)
    avatar_pos = new_state.get('avatar', [])[0]  # Access avatar position
    walls = new_state.get('wall', [])
    floors = new_state.get('floor', [])
    boxes = new_state.get('box', [])
    
    # Check if the action is a valid direction
    move = directions.get(action, [0, 0])  # Default to [0, 0] for 'noop' and invalid actions
    if move == [0, 0]:  # No operation or invalid action
        return new_state

    # Calculate new avatar position
    new_avatar_pos = [avatar_pos[0] + move[0], avatar_pos[1] + move[1]]
    
    # Check if the avatar's intended move is into a box
    if new_avatar_pos in boxes:
        # Calculate where the box would move
        box_index = boxes.index(new_avatar_pos)
        new_box_pos = [new_avatar_pos[0] + move[0], new_avatar_pos[1] + move[1]]
        
        # Check if the box can be moved (must remain on floors and not end up in walls or overlap another box)
        if (new_box_pos in floors and 
            new_box_pos not in walls and 
            new_box_pos not in boxes):
            # Update box position and move avatar
            boxes[box_index] = new_box_pos
            new_state['box'] = boxes
            new_state['avatar'] = [new_avatar_pos]
        else:
            # If box cannot be moved, avatar stays in the same position
            new_state['avatar'] = [avatar_pos]
    else:
        # If the new avatar position is within the floors and not in walls, update avatar position
        if new_avatar_pos in floors and new_avatar_pos not in walls:
            new_state['avatar'] = [new_avatar_pos]
        else:
            # Invalid move, avatar stays in the same position
            new_state['avatar'] = [avatar_pos]

    return new_state