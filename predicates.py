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


def is_on_border(state, loc):
    """
    Check if a given location is on the border of the map.

    Args:
    state (dict): The state dictionary containing entity positions.
    loc (list): The location to check (e.g., [0, 5]).

    Returns:
    bool: True if the location is on the border, False otherwise.
    """
    return loc in state.get('border', [])

def rule_formable(state, word1, word2, word3):
    """
    Check if the rule composed of word1, word2, word3 can be formed.

    Args:
    state (dict): The current game state.
    word1 (str): The first word in the rule.
    word2 (str): The second word in the rule.
    word3 (str): The third word in the rule.

    Returns:
    bool: True if the rule is formable, False otherwise.
    """
    for word in [word1, word2, word3]:
        if word.endswith('_word') and not pushable_word(state, word):
            return False
    return True


def pushable_word_up(state, word):
    """
    Check if the word can be pushed upwards.
    
    Args:
    state (dict): The current game state.
    word (str): The word entity to check.
    
    Returns:
    bool: True if the word is pushable upwards, False otherwise.
    """
    for (x, y) in state.get(word, []):
        obj_pos = [x, y + 1]      # Position where baba_obj needs to be to push up
        target_pos = [x, y - 1]   # Position where the word will be pushed to
        
        if target_pos not in state.get('empty', []):
            return False
        if obj_pos not in state.get('empty', []):
            return False
    return True

def pushable_word_down(state, word):
    """
    Check if the word can be pushed downwards.
    
    Args:
    state (dict): The current game state.
    word (str): The word entity to check.
    
    Returns:
    bool: True if the word is pushable downwards, False otherwise.
    """
    for (x, y) in state.get(word, []):
        obj_pos = [x, y - 1]      # Position where baba_obj needs to be to push down
        target_pos = [x, y + 1]   # Position where the word will be pushed to
        
        if target_pos not in state.get('empty', []):
            return False
        if obj_pos not in state.get('empty', []):
            return False
    return True

def pushable_word_left(state, word):
    """
    Check if the word can be pushed to the left.
    
    Args:
    state (dict): The current game state.
    word (str): The word entity to check.
    
    Returns:
    bool: True if the word is pushable to the left, False otherwise.
    """
    for (x, y) in state.get(word, []):
        obj_pos = [x + 1, y]      # Position where baba_obj needs to be to push left
        target_pos = [x - 1, y]   # Position where the word will be pushed to
        
        if target_pos not in state.get('empty', []):
            return False
        if obj_pos not in state.get('empty', []):
            return False
    return True

def pushable_word_right(state, word):
    """
    Check if the word can be pushed to the right.
    
    Args:
    state (dict): The current game state.
    word (str): The word entity to check.
    
    Returns:
    bool: True if the word is pushable to the right, False otherwise.
    """
    for (x, y) in state.get(word, []):
        obj_pos = [x - 1, y]      # Position where baba_obj needs to be to push right
        target_pos = [x + 1, y]   # Position where the word will be pushed to
        
        if target_pos not in state.get('empty', []):
            return False
        if obj_pos not in state.get('empty', []):
            return False
    return True


def pushable_word(state, word):
    """
    Check if the word is pushable in any direction.
    
    Args:
    state (dict): The current game state.
    word (str): The word entity to check.
    
    Returns:
    bool: True if the word is pushable in any direction, False otherwise.
    """
    return (
        pushable_word_up(state, word) or
        pushable_word_down(state, word) or
        pushable_word_left(state, word) or
        pushable_word_right(state, word)
    )

from itertools import product

def rule_breakable(state, word1, word2, word3):
    """
    Check if the rule composed of word1, word2, word3 is breakable.
    
    Args:
        state (dict): The current game state.
        word1 (str): The first word in the rule (e.g., "rock_word").
        word2 (str): The second word in the rule (e.g., "is_word").
        word3 (str): The third word in the rule (e.g., "flag_word").
    
    Returns:
        bool: True if the rule is breakable, False otherwise.
    """
    # First, check if the rule is formed
    if not rule_formed(state, word1, word2, word3):
        return False
    
    # Retrieve all coordinates for each word in the rule
    coords1 = state.get(word1, [])
    coords2 = state.get(word2, [])
    coords3 = state.get(word3, [])
    
    # Iterate through all possible triplet combinations
    for triplet in product(coords1, coords2, coords3):
        if are_adjacent(triplet):
            # For each word in the triplet, check if it's pushable in any direction
            for word, pos in zip([word1, word2, word3], triplet):
                if word.endswith('_word'):
                    x, y = pos
                    # Define possible directions
                    directions = {
                        'up':    ([x, y + 1], [x, y - 1]),
                        'down':  ([x, y - 1], [x, y + 1]),
                        'left':  ([x + 1, y], [x - 1, y]),
                        'right': ([x - 1, y], [x + 1, y])
                    }
                    # Check each direction for pushability
                    for direction, (obj_pos, target_pos) in directions.items():
                        if target_pos in state.get('empty', []) and obj_pos in state.get('empty', []):
                            # Found a pushable direction for this word instance
                            return True
    # If no pushable word instance is found in any triplet, the rule is not breakable
    return False

