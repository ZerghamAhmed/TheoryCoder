# make sure to include these import statements
from utils import directions

def transition_model(state, action):
    # Deep copy the state to avoid modifying the original state
    new_state = {
        'wall': [list(coord) for coord in state['wall']],
        'avatar': [list(coord) for coord in state['avatar']],
        'hole': [list(coord) for coord in state['hole']],
        'box': [list(coord) for coord in state['box']]
    }

    if action == 'noop':
        return new_state

    # Calculate new avatar position
    dx, dy = directions[action]
    new_avatar_pos = [new_state['avatar'][0][0] + dx, new_state['avatar'][0][1] + dy]

    # Check if the new avatar position is a wall
    if new_avatar_pos in new_state['wall']:
        return new_state  # Avatar can't move into a wall

    # Check if there's a box at the new avatar position
    if new_avatar_pos in new_state['box']:
        # Calculate new box position
        new_box_pos = [new_avatar_pos[0] + dx, new_avatar_pos[1] + dy]
        
        # Check if the new box position is a wall or another box
        if new_box_pos in new_state['wall'] or new_box_pos in new_state['box']:
            return new_state  # Box can't be pushed into a wall or another box

        # Move the box
        box_index = new_state['box'].index(new_avatar_pos)
        new_state['box'][box_index] = new_box_pos

    # Move the avatar
    new_state['avatar'][0] = new_avatar_pos

    return new_state
