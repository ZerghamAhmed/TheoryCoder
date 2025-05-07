# make sure to include these import statements
from utils import directions

def transition_model(state, action):
    # Get the current position of the avatar
    avatar_pos = state.get('avatar')[0]
    x, y = avatar_pos
    
    # Determine the direction of movement based on the action
    if action in directions:
        dx, dy = directions[action]
        new_x, new_y = x + dx, y + dy
        
        # Check if the new position is a wall
        if [new_x, new_y] in state.get('wall', []):
            # If there is a wall, the position doesn't change
            return state
        
        # Check if the new position is a trap
        if [new_x, new_y] in state.get('trap', []):
            # If there is a trap, the avatar "dies" - we can decide how to represent this
            # For now, let's assume the game ends or the avatar stays in place
            return state
        
        # Otherwise, move the avatar to the new position
        new_state = state.copy()
        new_state['avatar'] = [[new_x, new_y]]
        return new_state
    
    # If action is 'noop' or an invalid action, the state remains unchanged
    return state
