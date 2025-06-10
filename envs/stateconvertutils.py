def transform_coordinates(x, y, env_height):
    """
    Transform coordinates from top-left to bottom-left origin.
    """
    new_x = x
    # Correct the vertical mapping, ensuring bottom border is 0, and bottom playable row is 1
    new_y = env_height - y - 1
    return [new_x, new_y]






def convert_vgdl_state(env, previous_state=None):
    """
    Converts the VGDL state into a dictionary with keys:
    - Object positions grouped by type.
    - Avatar's carrying resources, direction, and pickaxe state.
    Ensures `avatar_carrying`, `avatar_direction`, and `avatar_pickaxe` are last.
    """
    full_state = env.Env.current_env._game.getFullState()
    objects = full_state['objects']
    block_size = env.Env.current_env._game.block_size
    env_height = env.Env.current_env._game.height


    state = {}


    # Process objects
    for obj_type, positions in objects.items():
        if obj_type == 'sword':  # Exclude sword as an independent object
            continue


        state[obj_type] = []
        for pos in positions.keys():
            # Ensure valid position
            if not isinstance(pos, tuple) or len(pos) != 2 or not all(isinstance(coord, int) for coord in pos):
                continue


            # Convert positions to grid coordinates
            x, y = pos
            grid_x = x // block_size
            grid_y = y // block_size
            grid_position = transform_coordinates(grid_x, grid_y, env_height)
            state[obj_type].append(grid_position)


    # Process avatar-related information
    avatar = env.Env.current_env._game.sprite_groups.get('avatar', [])
    if avatar:
        # Resources the avatar is carrying
        resources = env.Env.current_env._avatar.resources if env.Env.current_env._avatar else {}
        avatar_carrying = {resource: count for resource, count in resources.items() if count > 0}


        # Ensure diamonds key is always present
        if 'diamond' not in avatar_carrying:
            avatar_carrying['diamond'] = 0


        # Avatar direction (converted)
        orientation = env.Env.current_env._avatar.orientation if env.Env.current_env._avatar else (0, 0)
        avatar_direction = (
            orientation[0] // block_size,
            -orientation[1] // block_size  # Flip y-axis
        )


        # Avatar pickaxe state
        pickaxe_active = bool(objects.get('sword', []))
        if pickaxe_active:
            avatar_grid_pos = transform_coordinates(
                avatar[0].rect.left // block_size,
                avatar[0].rect.top // block_size,
                env_height
            )
            swing_position = [
                avatar_grid_pos[0] + avatar_direction[0],
                avatar_grid_pos[1] + avatar_direction[1]
            ]
        else:
            swing_position = []
    else:
        # Fallbacks for when the avatar is missing
        avatar_carrying = previous_state.get('avatar_carrying', {'diamond': 0}) if previous_state else {'diamond': 0}
        avatar_direction = previous_state.get('avatar_direction', (0, 0)) if previous_state else (0, 0)
        swing_position = []


    # Ensure avatar_pickaxe is initialized
    avatar_pickaxe = {
        'active': pickaxe_active if avatar else False,
        'swing_position': swing_position
    }


    # Append avatar-related keys at the end
    state['avatar_carrying'] = avatar_carrying
    state['avatar_direction'] = avatar_direction
    state['avatar_pickaxe'] = avatar_pickaxe


    return state




def convert_pb1_state_colorized(env, previous_state=None):
    """
    Converts the VGDL state into a dictionary with keys:
    - Object positions grouped by type.
    - Avatar's carrying resources, direction, and pickaxe state.
    Ensures `avatar_carrying`, `avatar_direction`, and `avatar_pickaxe` are last.
    """
    full_state = env.Env.current_env._game.getFullStateColorized()
    objects = full_state['objects']
    block_size = env.Env.current_env._game.block_size
    env_height = env.Env.current_env._game.height


    state = {}


    # Process objects
    for obj_type, positions in objects.items():


        state[obj_type] = []
        for pos in positions.keys():
            # Ensure valid position
            if not isinstance(pos, tuple) or len(pos) != 2 or not all(isinstance(coord, int) for coord in pos):
                continue


            # Convert positions to grid coordinates
            x, y = pos
            grid_x = x // block_size
            grid_y = y // block_size
            grid_position = transform_coordinates(grid_x, grid_y, env_height)
            state[obj_type].append(grid_position)


    # Process avatar-related information
    # avatar = env.Env.current_env._game.sprite_groups.get('DARKBLUE', [])


    return state


def convert_pb1_state(env, previous_state=None):
    """
    Converts the VGDL state into a dictionary with keys:
    - Object positions grouped by type.
    - Avatar's carrying resources, direction, and pickaxe state.
    Ensures `avatar_carrying`, `avatar_direction`, and `avatar_pickaxe` are last.
    """
    full_state = env.Env.current_env._game.getFullState()
    objects = full_state['objects']
    block_size = env.Env.current_env._game.block_size
    env_height = env.Env.current_env._game.height


    state = {}


    # Process objects
    for obj_type, positions in objects.items():


        state[obj_type] = []
        for pos in positions.keys():
            # Ensure valid position
            if not isinstance(pos, tuple) or len(pos) != 2 or not all(isinstance(coord, int) for coord in pos):
                continue


            # Convert positions to grid coordinates
            x, y = pos
            grid_x = x // block_size
            grid_y = y // block_size
            grid_position = transform_coordinates(grid_x, grid_y, env_height)
            state[obj_type].append(grid_position)


    # Process avatar-related information
    # avatar = env.Env.current_env._game.sprite_groups.get('DARKBLUE', [])


    return state


