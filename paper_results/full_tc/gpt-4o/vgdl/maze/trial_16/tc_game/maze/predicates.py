def avatargoal(state, avatar, goal):
    """
    Checks if the avatar and goal positions overlap.

    Parameters:
    - state: dict with keys as object names and values as lists of positions
    - avatar: object name (e.g., 'avatar')
    - goal: object name (e.g., 'goal')

    Returns:
    - bool: True if the avatar's position overlaps with the goal's position, False otherwise
    """
    avatar_pos = state.get(avatar, [])
    goal_pos = state.get(goal, [])
    if not avatar_pos or not goal_pos:
        return False
    return avatar_pos[0] == goal_pos[0]