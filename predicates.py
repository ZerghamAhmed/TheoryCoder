from itertools import product
from collections import deque

def negate(result):
    """Return the negated result of the given value."""
    return not result

def are_adjacent(coords):
    """Check if the coordinates are adjacent horizontally or vertically."""
    if len(coords) != 3:
        return False

    # print("Coordinate Combos:", coords)

    # Check horizontal adjacency
    if coords[0][1] == coords[1][1] == coords[2][1] and coords[1][0] == coords[0][0] + 1 and coords[2][0] == coords[1][0] + 1:
        print('horizontal adjacent')
        return True

    # Check vertical adjacency
    if coords[0][0] == coords[1][0] == coords[2][0] and coords[1][1] == coords[0][1] - 1 and coords[2][1] == coords[1][1] - 1:
        print('vertical adjacent')
        return True

    return False

def rule_formed(state, word1, word2, word3):
    """Check if the given words are adjacent in the state."""
    coords1 = state.get(word1, [])
    coords2 = state.get(word2, [])
    coords3 = state.get(word3, [])

    if not coords1 or not coords2 or not coords3:
        return False

    # Generate all possible triplets of coordinates, ensuring each word is used once
    for triplet in product(coords1, coords2, coords3):
        if are_adjacent(list(triplet)):
            return True

    return False

# def at(state, entity, loc):
#     """
#     Check if the specific instance of an entity is at the given location.

#     Args:
#     state (dict): The state dictionary containing entity positions.
#     entity (str): The entity to check (e.g., "flag_word").
#     loc (list): The location to check (e.g., [6, 8]). MUST BE A LIST NOT Tuple.

#     Returns:
#     bool: True if the entity is at the location, False otherwise.
#     """
#     # Get the list of coordinates for the entity
#     coords = state.get(entity, [])
#     # Ensure loc 
#     if list(loc) in coords:
#         return True

#     # Check if the location is in the list of coordinates
#     return False

def overlapping(state, entity1, index1, entity2, index2):
    """
    Check if a specific instance of one entity overlaps (shares the same coordinate) with a specific instance of another entity.

    Args:
    state (dict): The state dictionary containing entity positions.
    entity1 (str): The first entity to check.
    index1 (int): The index of the instance of the first entity.
    entity2 (str): The second entity to check.
    index2 (int): The index of the instance of the second entity.

    Returns:
    bool: True if the specified instances overlap, False otherwise.
    """
    # Get the list of coordinates for both entities
    coords1 = state.get(entity1, [])
    coords2 = state.get(entity2, [])

    # Check if the indices are within the bounds of the coordinate lists
    if index1 < len(coords1) and index2 < len(coords2):
        # Compare the coordinates at the specified indices
        return tuple(coords1[index1]) == tuple(coords2[index2])

    return False


def at(state, entity, loc, index=None):
    """
    Check if the specific instance of an entity is at the given location.
    If index is None, check if any instance of the entity is at the location.

    Args:
    state (dict): The state dictionary containing entity positions.
    entity (str): The entity to check (e.g., "flag_word").
    loc (list): The location to check (e.g., [6, 8]). MUST BE A LIST NOT Tuple.
    index (int, optional): The index of the specific instance to check. Defaults to None.

    Returns:
    bool: True if the entity (or specific instance) is at the location, False otherwise.
    """
    # breakpoint()
    # Get the list of coordinates for the entity
    coords = state.get(entity, [])

    # breakpoint()
    # breakpoint()
    # Check if a specific instance is requested
    if index is not None:
        if 0 <= index < len(coords):
            return loc == coords[index]
        else:
            return False

    # breakpoint()

    # Check if the location is in the list of coordinates for any instance
    return loc in coords