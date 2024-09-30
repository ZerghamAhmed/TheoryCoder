from predicates import *
from copy import deepcopy
from games import BabaIsYou
from babareport import BabaReportUpdater
from utils import directions

# Function to update pushable entity's coordinates
def make_push(state, pushable, old_coords, delta):
    if pushable in state:
        pushable_new_coords = [old_coords[0] + delta[0], old_coords[1] + delta[1]] 
        index = state[pushable].index(old_coords)
        state[pushable][index] = pushable_new_coords
        print(f"PUSHED {pushable} from {old_coords} to {pushable_new_coords}")

def simulate_movement(new_coordinates, delta, grid_size=10):
    listofcoords = []
    for i in range(1, grid_size + 1):
        next_coords = [new_coordinates[0] + i * delta[0], new_coordinates[1] + i * delta[1]]
        listofcoords.append(next_coords)
    return listofcoords

# make sure to include these import statements
from predicates import *
from copy import deepcopy
from games import BabaIsYou
from babareport import BabaReportUpdater
from utils import directions

def transition_model(state, action):
    # Apply object transformation rules
    transformations = {}  # Mapping from old_obj to new_obj

    object_words = {entity for entity in state if entity.endswith('_word') and entity not in ['is_word', 'push_word', 'you_word', 'win_word', 'stop_word']}

    for obj_word1 in object_words:
        for obj_word2 in object_words:
            if obj_word1 != obj_word2:
                if rule_formed(state, obj_word1, 'is_word', obj_word2):
                    old_obj = obj_word1[:-5] + '_obj'
                    new_obj = obj_word2[:-5] + '_obj'
                    # breakpoint()

                    transformations[old_obj] = new_obj

    # Apply the transformations
    entities_to_remove = set()
    entities_to_add = {}

    for old_obj, new_obj in transformations.items():
        if old_obj in state:
            coords = state[old_obj]
            entities_to_remove.add(old_obj)
            if new_obj not in entities_to_add:
                entities_to_add[new_obj] = []
            entities_to_add[new_obj].extend(coords)

    # Remove the old objects
    for entity in entities_to_remove:
        del state[entity]

    # Add the new objects
    for entity, coords in entities_to_add.items():
        if entity not in state:
            state[entity] = []
        state[entity].extend(coords)

    # Determine controllable entities (objects corresponding to "is you" rule)
    controllables = {
        entity for entity in state
        if rule_formed(state, f'{entity[:-4]}_word', 'is_word', 'you_word')
    }
    print("controllables", controllables)

    # Determine initial pushable entities
    pushables = {
        entity for entity in state
        if (
            (entity.endswith('_obj') and rule_formed(state, f'{entity[:-4]}_word', 'is_word', 'push_word'))
            or entity.endswith('_word')
        )
    }
    print("initial pushables", pushables)

    # Determine 'stop' entities (objects corresponding to "is stop" rule)
    stop_entities = set()
    for entity in state:
        if entity.endswith('_obj'):
            word_entity = f'{entity[:-4]}_word'
            if word_entity in state and rule_formed(state, word_entity, 'is_word', 'stop_word'):
                stop_entities.add(entity)
    print("stop entities", stop_entities)

    # Build obstacle_coords from 'stop' entities
    obstacle_coords = set()
    for entity in stop_entities:
        for coord in state[entity]:
            obstacle_coords.add(tuple(coord))
    print("obstacle_coords", obstacle_coords)

    pushables_coords = set()
    for pushable in pushables:
        for coord in state[pushable]:
            pushables_coords.add(tuple(coord))

    empties_coords = set()
    for coord in state["empty"]:
        empties_coords.add(tuple(coord))

    # get controllable entity
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

                        if tuple(new_coords) in obstacle_coords:
                            # Cannot move into a 'Stop' entity
                            return state

                        if tuple(new_coords) not in pushables_coords:

                            # update obj coordinates after move
                            state[obj][i] = new_coords

                            if tuple(new_coords) in empties_coords:
                                state["empty"].remove(new_coords)

                            state["empty"].append(coord)

                        # will not change coordinate
                        elif tuple(new_coords) in pushables_coords:
                            stack = []

                            if delta == [1, 0]:  # right
                                rest_grid = 9 - coord[0]
                            elif delta == [-1, 0]:
                                rest_grid = coord[0]
                            elif delta == [0, 1]:  # up
                                rest_grid = 9 - coord[1]
                            elif delta == [0, -1]:
                                rest_grid = coord[1]
                            else:
                                print("ERROR IN DELTA")

                            # see coordinates along direction vector
                            rest_of_coords = simulate_movement(coord, delta, grid_size=rest_grid)
                            available_space = False
                            blocked_space = False

                            for square in rest_of_coords:
                                if tuple(square) in obstacle_coords:
                                    # Cannot push into or through a 'Stop' entity
                                    blocked_space = True
                                    break
                                elif tuple(square) in pushables_coords:
                                    # Check if the pushable entity at this coordinate is 'Stop'
                                    pushable_entity = None
                                    for pushable in pushables:
                                        if square in state[pushable]:
                                            pushable_entity = pushable
                                            break
                                    if pushable_entity in stop_entities:
                                        # Cannot push a 'Stop' entity
                                        blocked_space = True
                                        break
                                    else:
                                        stack.append(square)
                                elif tuple(square) in empties_coords:
                                    available_space = True
                                    break

                            print(f"Stack after simulation: {stack}")

                            if available_space and not blocked_space:
                                print("THERE IS SPACE FOR PUSHING")

                                stack.reverse()

                                # entities to be pushed
                                print(stack)
                                for entity_coords in stack:
                                    for pushable in pushables:
                                        if entity_coords in state[pushable]:
                                            make_push(state, pushable, entity_coords, delta)

                                # controlled object will be moved to spot and push others of Stack to their new coords
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
                                return state

    return state
