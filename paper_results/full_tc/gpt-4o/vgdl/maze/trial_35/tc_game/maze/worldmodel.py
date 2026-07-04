# make sure to include these import statements
from utils import directions

def transition_model(state, action):
    current_state = state.copy()
    avatar_pos = current_state.get('avatar', [])[0]  # Access avatar position
    walls = current_state.get('wall', [])
    floors = current_state.get('floor', [])

    # Calculate the new avatar position based on the action
    if action in directions:
        move = directions[action]
        new_avatar_pos = [avatar_pos[0] + move[0], avatar_pos[1] + move[1]]

        # Check if the new position is valid (i.e., within floor and not in walls)
        if new_avatar_pos in floors and new_avatar_pos not in walls:
            current_state['avatar'] = [new_avatar_pos]  # Update avatar position
        else:
            # If invalid move (e.g., wall or out of bounds), avatar stays in the same position
            current_state['avatar'] = [avatar_pos]
    else:
        # 'noop' action or invalid action, no change to avatar position
        current_state['avatar'] = [avatar_pos]

    return current_state