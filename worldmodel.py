from predicates import *
from copy import deepcopy
from games import BabaIsYou

def print_formed_rules(state):
    word_entities = [entity for entity in state.keys() if entity.endswith('_word')]
    rules_on_map = []
    for subj in word_entities:
        for pred in word_entities:
            for obj in word_entities:
                if rule_formed(state, subj, pred, obj):
                    print(f"Rule formed: {subj} {pred} {obj}")
                    rules_on_map.append({'subject': subj, 'predicate': pred, 'object': obj})
    return rules_on_map

def apply_transmutations(state, rules):
    non_transmutation_objects = {'you_word', 'win_word', 'push_word', 'sink_word', 'kill_word', 'melt_word', 'hot_word', 'move_word'}
    transmutation_rules = [
        rule for rule in rules if rule['object'] not in non_transmutation_objects
    ]
    
    for rule in transmutation_rules:
        subject_obj = rule['subject'].replace('_word', '_obj')
        object_obj = rule['object'].replace('_word', '_obj')
        if subject_obj in state:
            state[object_obj] = state.pop(subject_obj)
            print(f"Transmuted {subject_obj} to {object_obj}")

    return state

# Function to update pushable entity's coordinates
def make_push(state, pushable, old_coords, delta):
    print(state[pushable])
    if pushable in state:
        pushable_new_coords = [old_coords[0] + delta[0], old_coords[1] + delta[1]]  
        print(old_coords, "are OLD coords")
        print(pushable_new_coords, "are the new coords")
        index = state[pushable].index(list(old_coords))
        state[pushable][index] = list(pushable_new_coords)
        print(f"PUSHED {pushable} from {old_coords} to {pushable_new_coords}")

def simulate_movement(new_coordinates, delta, grid_size=10):
    listofcoords = []
    for i in range(1, grid_size + 1):
        next_coords = [new_coordinates[0] + i * delta[0], new_coordinates[1] + i * delta[1]]
        listofcoords.append(next_coords)
    return listofcoords


def transition_model(state, action):
    directions = {
        'left': [-1, 0],
        'right': [1, 0],
        'up': [0, 1],
        'down': [0, -1]
    }

    active_rules = print_formed_rules(state)  # Print all rules formed at the beginning
    state = apply_transmutations(state, active_rules)  # Apply transmutations

    # Determine controllable entities (objects corresponding to "is you" rule)
    controllables = {
        entity for entity in state
        if rule_formed(state, f'{entity[:-4]}_word', 'is_word', 'you_word')
    }
    print("controllables", controllables)


    # Determine automover entities (objects corresponding to "is move" rule)
    automovers = {
        entity for entity in state
        if rule_formed(state, f'{entity[:-4]}_word', 'is_word', 'move_word')
    }
    print("automovers", automovers)

    # Determine controllable entities (objects corresponding to "is you" rule)
    meltables = {
        entity for entity in state
        if rule_formed(state, f'{entity[:-4]}_word', 'is_word', 'hot_word')
    }
    print("meltables", meltables)

    # Determine entities that will kill you (objects corresponding to "is kill" rule)
    killables = {
        entity for entity in state
        if rule_formed(state, f'{entity[:-4]}_word', 'is_word', 'kill_word')
    }
    print("killables", killables)

    # Determine entities that will sink you (objects corresponding to "is sink" rule)
    sinkables = {
        entity for entity in state
        if rule_formed(state, f'{entity[:-4]}_word', 'is_word', 'sink_word')
    }
    print("sinkables", sinkables)

    # Determine initial pushable entities
    pushables = {
        entity for entity in state
        if (
            (entity.endswith('_obj') and rule_formed(state, f'{entity[:-4]}_word', 'is_word', 'push_word'))
            or entity.endswith('_word')
        )
    }
    print("initial pushables", pushables)

    # Determine obstacles, excluding controllables and pushables
    obstacles = {
        entity for entity in state
        if entity not in pushables and entity not in controllables
        and entity not in ['empty', 'lost', 'won'] 
    }
    print("obstacles:", obstacles)

    # Remove entities that are part of the win rule, so you are able to overlap with them and they are not considered obstacles
    for entity in state:
        if entity.endswith('_word') and rule_formed(state, f'{entity}', 'is_word', 'win_word'):
            obj_entity = entity.replace('_word', '_obj')
            obstacles.discard(obj_entity)  

    print("filtered obstacles", obstacles)

    # Determine winners
    winables = {
        entity.replace('_word', '_obj') for entity in state
        if entity.endswith('_word') and rule_formed(state, f'{entity}', 'is_word', 'win_word')
    }
    print("winners:", winables)

    # Collect winner coordinates
    win_coords = set()
    for winner in winables:
        if winner in state:
            for coord in state[winner]:
                win_coords.add(tuple(coord))

    print("win_coords:", win_coords)

    overlapables = set()
    # Check if previously killable or sinkable entities should now be moved to overlaps
    for entity in state:
        if entity.endswith('_obj') and entity not in winables and entity not in controllables and entity not in pushables:
            if not rule_formed(state, entity.replace('_obj', '_word'), 'is_word', 'kill_word') and not rule_formed(state, entity.replace('_obj', '_word'), 'is_word', 'sink_word'):
                overlapables.add(entity.replace('_word', '_obj'))
                obstacles.discard(entity.replace('_word', '_obj'))  # Remove as obstacles and kill zone
            elif rule_formed(state, entity.replace('_obj', '_word'), 'is_word', 'sink_word'):
                obstacles.discard(entity.replace('_word', '_obj'))  # Remove as obstacles and kill zone

    print("overlapables:", overlapables)

    overlapables_coords = set()
    for overlapable in overlapables:
        for coord in state[overlapable]:
            overlapables_coords.add(tuple(coord)) 

    obstacle_coords = set()
    for obstacle in obstacles:
        for coord in state[obstacle]:
            obstacle_coords.add(tuple(coord))

    pushables_coords = set()
    for pushable in pushables:
        for coord in state[pushable]:
            pushables_coords.add(tuple(coord))

    print("pushable coords", pushables_coords)

    empties_coords = set()
    for coord in state["empty"]:
        empties_coords.add(tuple(coord))  

    print("ARE EMPTY", empties_coords)

    print("OG STATE EMPTY", state["empty"])

    killables_coords = set()
    for killable in killables:
        for coord in state[killable]:
            killables_coords.add(tuple(coord))  

    sinkables_coords = set()
    for sinkable in sinkables:
        for coord in state[sinkable]:
            sinkables_coords.add(tuple(coord))  

    if automovers:
        directions = {
        'left': [-1, 0],
        'right': [1, 0],
        'up': [0, 1],
        'down': [0, -1],
        'stop': [0, 0]
    }

    # looks for automover entity and focuses on moving instance of that entity
    for obj, coords in state.items():
        if obj.endswith('_obj'):
            obj_name = obj[:-4]
            if obj in automovers and action == "stop":

                print(f"AUTOMOVER {obj_name}_obj")

                # turns action into coordinate change for state dict
                if action in directions:
                    delta = [1,0]

                    # all automovers will move right until they collide with anything then they will move left until they collide, move right until collide, repeat
                    for i, coord in enumerate(coords):
                        print("printing coords", coord)
                        new_coords = [coord[0] + delta[0], coord[1] + delta[1]]
                        print('new coordinates to move to ', new_coords)
                        if tuple(new_coords) not in obstacle_coords and tuple(new_coords) not in pushables_coords and tuple(new_coords) not in sinkables_coords:
                            print("MOVING WITHOUT OBSTACLES")
                            print("ACTION RESTATE", action)
                            state[obj][i] = new_coords
                            print("new state empty", state["empty"])
                            print("empties coords", empties_coords)

                            if tuple(new_coords) in empties_coords:
                                state["empty"].remove(new_coords)

                            state["empty"].append(coord)

                        elif tuple(new_coords) in obstacle_coords:
                            return state

                        elif tuple(new_coords) in killables_coords or tuple(new_coords) in sinkables_coords:
                            state["lost"] = True
                            return state

                        # make sure the move is actually valid (not in border, no obstruction) 
                        elif tuple(new_coords) in pushables_coords:
                            stack = []

                            if delta == [1, 0]:  # right
                                rest_grid = 9 - coord[0]
                            elif delta == [-1, 0]:  # left
                                rest_grid = coord[0]
                            elif delta == [0, 1]: # up
                                rest_grid = 9 - coord[1]
                            elif delta == [0, -1]: # down
                                rest_grid = coord[1]
                            else:
                                print("ERROR IN DELTA")

                            # see coordinates along direction vector
                            rest_of_coords = simulate_movement(coord, delta, grid_size=rest_grid)

                            print("REST OF COORDS:", rest_of_coords)

                            available_space = None
                            blocked_space = None
                            for square in rest_of_coords:
                                print("SQUARE", square)
                                print(pushables_coords)
                                if tuple(square) in pushables_coords:
                                    stack.append(square)
                                    print(stack)
                                elif tuple(square) in empties_coords:
                                    available_space = True
                                    break
                                elif tuple(square) in obstacle_coords:
                                    blocked_space = True
                                    break

                            print(f"Stack after simulation: {stack}")

                            if available_space:
                                print("THERE IS SPACE FOR PUSHING")
                                stack.reverse()
                                print(stack)
                                for entity_coords in stack:
                                    for pushable in pushables:
                                        if entity_coords in state[pushable]:
                                            new_pushed_coords = [entity_coords[0] + delta[0], entity_coords[1] + delta[1]]
                                            make_push(state, pushable, entity_coords, delta)

                                            # sink mechanic
                                            if tuple(new_pushed_coords) in sinkables_coords:
                                                print(f"{pushable} sank in goop at {new_pushed_coords}")
                                                state["empty"].append(new_pushed_coords)

                                                # remove the pushable that ended up in the goop after make_push 
                                                state[pushable].remove(new_pushed_coords)

                                                for goop in sinkables:
                                                    if new_pushed_coords in state[goop]:
                                                        # remove that goop obj that was at the location being pushed to
                                                        state[goop].remove(new_pushed_coords)
                                                    
                                # baba will be moved to spot 1 and push others of Stack to their new coords
                                state[obj][i] = new_coords
                                state["empty"].append(coord)

                                # Update the empty coordinates by removing the first coordinate in the stack
                                first_in_stack = stack[0]
                                if tuple(first_in_stack) in empties_coords:
                                    state["empty"].remove(first_in_stack)
                                    
                                else:
                                    state["empty"].append(first_in_stack)
                                    return state

                            else:
                                print("MOVE BLOCKED")
                                return state

    # looks for controllable entity and focuses on moving instance of that entity
    for obj, coords in state.items():
        if obj.endswith('_obj'):
            obj_name = obj[:-4]
            if obj in controllables:
                print(f"Controlling {obj_name}_obj")

                # turns action into coordinate change for state dict
                if action in directions:
                    delta = directions[action]

                    # need to move all controlled entities in a given direction
                    for i, coord in enumerate(coords):
                        print("printing coords", coord)

                        # calculate new coordinates after move
                        new_coords = [coord[0] + delta[0], coord[1] + delta[1]]
                        print('new coordinates to move to ', new_coords)

                        # make sure the move is valid (not in border, no obstruction) 
                        if tuple(new_coords) not in obstacle_coords and tuple(new_coords) not in pushables_coords and tuple(new_coords) not in sinkables_coords:
                            print("MOVING WITHOUT OBSTACLES")
                            print("ACTION RESTATE", action)

                        # update obj coordinates after move
                            state[obj][i] = new_coords

                            print("new state empty", state["empty"])
                            print("empties coords", empties_coords)

                            if tuple(new_coords) in empties_coords:
                                state["empty"].remove(new_coords)

                            state["empty"].append(coord)

                         # will not change coordinate since can't 
                        elif tuple(new_coords) in obstacle_coords:
                            return state

                        elif tuple(new_coords) in killables_coords or tuple(new_coords) in sinkables_coords:
                            state["lost"] = True
                            return state

                        # # make sure the move is valid (not in border, no obstruction) 
                        elif tuple(new_coords) in pushables_coords:
                            stack = []

                            if delta == [1, 0]:  # right
                                rest_grid = 9 - coord[0]
                            elif delta == [-1, 0]:   
                                rest_grid = coord[0]
                            elif delta == [0, 1]: # up
                                rest_grid = 9 - coord[1]
                            elif delta == [0, -1]:
                                rest_grid = coord[1]
                            else:
                                print("ERROR IN DELTA")

                            # see coordinates along direction vector
                            rest_of_coords = simulate_movement(coord, delta, grid_size=rest_grid)

                            print("REST OF COORDS:", rest_of_coords)

                            available_space = None
                            blocked_space = None
                            for square in rest_of_coords:
                                print("SQUARE", square)
                                print(pushables_coords)
                                if tuple(square) in pushables_coords:
                                    stack.append(square)
                                    print(stack)
                                elif tuple(square) in empties_coords:
                                    available_space = True
                                    break
                                elif tuple(square) in obstacle_coords:
                                    blocked_space = True
                                    break

                            print(f"Stack after simulation: {stack}")

                            if available_space:
                                print("THERE IS SPACE FOR PUSHING")
                                stack.reverse()
                                print(stack)
                                for entity_coords in stack:
                                    for pushable in pushables:
                                        if entity_coords in state[pushable]:
                                            new_pushed_coords = [entity_coords[0] + delta[0], entity_coords[1] + delta[1]]

                                            make_push(state, pushable, entity_coords, delta)

                                            # sink mechanic
                                            if tuple(new_pushed_coords) in sinkables_coords:
                                                print(f"{pushable} sank in goop at {new_pushed_coords}")
                                                state["empty"].append(new_pushed_coords)

                                                # remove the pushable that ended up in the goop after make_push 
                                                state[pushable].remove(new_pushed_coords)

                                                for goop in sinkables:
                                                    if new_pushed_coords in state[goop]:
                                                        # remove that goop obj that was at the location being pushed to
                                                        state[goop].remove(new_pushed_coords)

                                # baba will be moved to spot 1 and push others of Stack to their new coords
                                state[obj][i] = new_coords
                                state["empty"].append(coord)
                               
                                # Update the empty coordinates by removing the first coordinate in the stack
                                first_in_stack = stack[0]
                                if tuple(first_in_stack) in empties_coords:
                                    state["empty"].remove(first_in_stack)
                                    
                                else:
                                    state["empty"].append(first_in_stack)
                                    return state

                            else:
                                print("MOVE BLOCKED")
                                return state
                                          
    return state

