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
        # print('horizontal adjacent')
        return True

    # Check vertical adjacency
    if coords[0][0] == coords[1][0] == coords[2][0] and coords[1][1] == coords[0][1] - 1 and coords[2][1] == coords[1][1] - 1:
        # print('vertical adjacent')
        return True

    return False

def rule_formed(state, word1, word2, word3):
    """
    Determine if a valid rule is formed by the three specified words in the given state.

    A rule is formed if:
    - The three words appear in the state with valid coordinates.
    - The words are aligned either horizontally or vertically.
    - The words are adjacent to each other with no gaps.
    - The second word is 'is_word' (a requirement for rule formation).

    Args:
        state (dict): A dictionary representing the current game state. 
                      Keys are word names (e.g., "baba_word") mapped to lists of their coordinates.
        word1 (str): The first word in the rule (e.g., "baba_word").
        word2 (str): The second word in the rule (must be "is_word").
        word3 (str): The third word in the rule (e.g., "win_word").

    Returns:
        bool: True if the three words form a valid rule, False otherwise.

    Example:
        state = {
            "baba_word": [[0, 0]],
            "is_word": [[0, 1]],
            "you_word": [[0, 2]],
        }
        rule_formed(state, "baba_word", "is_word", "you_word")  # Returns True, as the words are horizontally adjacent.

    Notes:
        - This function uses the `are_adjacent` helper to verify adjacency and alignment of the words.
        - If any of the words are not present in the state or do not have coordinates, the function returns False.
    """
    coords1 = state.get(word1, [])
    coords2 = state.get(word2, [])
    coords3 = state.get(word3, [])

    if not coords1 or not coords2 or not coords3:
        return False
    
    if word2 != 'is_word':
        return False

    # Generate all possible triplets of coordinates, ensuring each word is used once
    for triplet in product(coords1, coords2, coords3):
        if are_adjacent(list(triplet)):
            return True

    return False

def overlapping(state, entity1, entity2):
    """
    Check if any instance of one entity overlaps (shares the same coordinate) with any instance of another entity.

    Args:
    state (dict): The state dictionary containing entity positions.
    entity1 (str): The first entity to check.
    entity2 (str): The second entity to check.

    Returns:
    bool: True if any instances overlap, False otherwise.
    """
    coords1 = state.get(entity1, [])
    coords2 = state.get(entity2, [])

    for coord1 in coords1:
        if coord1 in coords2:
            return True

    return False

def at(state, entity, loc):
    """
    Check if any instance of an entity is at the given location.

    Args:
    state (dict): The state dictionary containing entity positions.
    entity (str): The entity to check (e.g., "flag_word").
    loc (list): The location to check (e.g., [6, 8]).

    Returns:
    bool: True if any instance of the entity is at the location, False otherwise.
    """
    coords = state.get(entity, [])
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
    Check if the rule composed of word1, word2, word3 is formable.
    
    A rule is formable if:
    - It is not already formed.
    - None of the specific instances of the words in the rule are reused within the same rule.
    - word1 and word3 are not part of an unbreakable rule (where rule_breakable = False).
    
    Args:
        state (dict): The current game state.
        word1 (str): The first word in the rule (e.g., "baba_word").
        word2 (str): The second word in the rule (e.g., "is_word").
        word3 (str): The third word in the rule (e.g., "win_word").
    
    Returns:
        bool: True if the rule is formable, False otherwise.
    """

    # Words that cannot be used to start a rule
    end_words = {
        'you_word',
        'win_word',
        'push_word',
        'stop_word',
        'hot_word',
        'melt_word',
        'sink_word',
        'kill_word',
    }
    
    # Step 1: Check if the rule is already formed
    if rule_formed(state, word1, word2, word3):
        return False  # Rule is already formed; cannot be formed again
    
    # Step 2: If word1 is an end word or 'is_word', it cannot form a new rule
    if word1 in end_words or word1 == "is_word" or word3 == "is_word" or word2 != 'is_word':
        return False

    # Step 3: Get the coordinates of the words in the state
    coords1 = state.get(word1, [])
    coords2 = state.get(word2, [])
    coords3 = state.get(word3, [])
    
    if len(coords1) == 0 or len(coords2) == 0 or len(coords3) == 0:
        return False  # If any word has no coordinates, the rule cannot be formed

    # Step 4: Find all unbreakable rules and track their word instances (coordinates)
    def find_unbreakable_rules(state):
        """
        Find all unbreakable rules on the map and return the word instances (coordinates) involved.
        
        Args:
            state (dict): The current game state.
        
        Returns:
            set: A set of (word, coord) tuples that represent the instances in unbreakable rules.
        """
        unbreakable_instances = set()
        word_entities = [entity for entity in state.keys() if entity.endswith('_word')]
        
        # Loop over all possible combinations of subject, predicate, and object
        for subj in word_entities:
            for pred in word_entities:
                for obj in word_entities:
                    if rule_formed(state, subj, pred, obj):
                        if not rule_breakable(state, subj, pred, obj):
                            # Add the specific word instances (coordinates) involved in unbreakable rules
                            for coord1, coord2, coord3 in product(state.get(subj, []), state.get(pred, []), state.get(obj, [])):
                                if are_adjacent((coord1, coord2, coord3)):  # Ensure they are adjacent
                                    unbreakable_instances.add((subj, tuple(coord1)))
                                    unbreakable_instances.add((obj, tuple(coord3)))
        return unbreakable_instances

    # Step 5: Check if any specific instances of word1 or word3 are part of any unbreakable rule
    unbreakable_instances = find_unbreakable_rules(state)
    
    # Allow only the instances that are not part of an unbreakable rule
    for coord1, coord3 in product(coords1, coords3):
        if (word1, tuple(coord1)) in unbreakable_instances or (word3, tuple(coord3)) in unbreakable_instances:
            continue  # Skip this instance if it's part of an unbreakable rule
        # Valid instance found, break out of loop and form the rule.
        break
    else:
        return False  # No valid instances, rule cannot be formed

    # Step 6: Prevent rules like 'baba_word is_word baba_word' if only one instance of the word exists
    if (len(coords1) == 1 and word1 == word3) or (len(coords3) == 1 and word1 == word3):
        return False  # If word1 and word3 are the same and only one instance, block the rule
    
    # Step 7: Ensure that no word instance is reused within the same rule
    for coord1, coord2, coord3 in product(coords1, coords2, coords3):
        # Check if any of the coordinates are the same (i.e., word reuse in the same rule)
        if coord1 == coord2 or coord1 == coord3 or coord2 == coord3:
            return False  # Invalid because the same instance of a word is reused

    if not pushable_word(state, word1) and not pushable_word(state, word2) and pushable_word(state, word3):
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

def get_all_word_entities(state):
    """
    Extract all relevant word entities from the current game state.
    
    Args:
        state (dict): The current game state.
    
    Returns:
        list: A list of all word entities (e.g., "baba_word", "is_word", "flag_word").
    """
    word_entities = [key for key in state if key.endswith('_word')]
    return word_entities

def generate_potential_rules(state):
    """
    Generate all potential rules based on the current game state.
    
    Args:
        state (dict): The current game state.
    
    Returns:
        list of tuples: A list of potential rules (word1, word2, word3).
    """
    word_entities = get_all_word_entities(state)
    
    potential_rules = []
    for word1, word3 in product(word_entities, repeat=2):
        if word1 != word3:  # Avoid self-rules like 'baba_word is_word baba_word'
            potential_rules.append((word1, 'is_word', word3))
    
    return potential_rules
def get_unbreakable_rule_instances(state):
    """
    Retrieve all unbreakable rule instances (specific coordinates) from the current game state.
    
    A rule is considered unbreakable if:
    - It is currently formed.
    - It is not breakable (rule_breakable returns False).
    
    Args:
        state (dict): The current game state.
    
    Returns:
        list of tuples: Each tuple represents an unbreakable rule with specific word coordinates
                        in the format ((word1_coords), (word2_coords), (word3_coords)).
    """
    unbreakable_instances = []
    
    # Generate all potential rules dynamically
    potential_rules = generate_potential_rules(state)
    
    # Iterate through each potential rule
    for word1, word2, word3 in potential_rules:
        if rule_formed(state, word1, word2, word3) and not rule_breakable(state, word1, word2, word3):
            # Retrieve the specific coordinates (instances) of the words in the unbreakable rule
            coords1 = state.get(word1, [])
            coords2 = state.get(word2, [])
            coords3 = state.get(word3, [])
            
            # Identify the triplet of coordinates that form the rule
            for triplet in product(coords1, coords2, coords3):
                if are_adjacent(triplet):
                    # Add this specific instance of the unbreakable rule
                    unbreakable_instances.append(triplet)
    
    return unbreakable_instances


def get_unbreakable_rules(state):
    """
    Retrieve all unbreakable rules from the current game state.
    
    A rule is considered unbreakable if:
    - It is currently formed.
    - It is not breakable (rule_breakable returns False).
    
    Args:
        state (dict): The current game state.
    
    Returns:
        list of tuples: A list of unbreakable rules, where each rule is a tuple (word1, word2, word3).
    """
    unbreakable_rules = []
    
    # Generate all potential rules dynamically
    potential_rules = generate_potential_rules(state)
    
    # Iterate through each potential rule
    for rule in potential_rules:
        word1, word2, word3 = rule
        
        # Check if the rule is formed and is unbreakable
        if rule_formed(state, word1, word2, word3) and not rule_breakable(state, word1, word2, word3):
            # Add the unbreakable rule to the list
            unbreakable_rules.append((word1, word2, word3))

    if state["flag_word"] == [[6,5]]:
        breakpoint()
    return unbreakable_rules


def overlapable(state, entity):
    """
    Check if the given entity is overlapable.

    Args:
        state (dict): The current game state.
        entity (str): The entity to check (e.g., "baba_obj", "flag_obj").

    Returns:
        bool: True if the entity ends with '_obj', False otherwise.
    """
    return entity.endswith('_obj')

def pushable_obj(state, entity):
    """
    Check if the given object entity is pushable based on the rule 'X_word is_word push_word'.

    Args:
        state (dict): The current game state.
        entity (str): The object entity to check (e.g., "baba_obj", "rock_obj").

    Returns:
        bool: True if the corresponding rule 'X_word is_word push_word' is formed, False otherwise.

    Example: 
        If 'rock_word is_word push_word' is formed, then 'rock_obj' is pushable_obj.
    """
    # Step 1: Ensure the entity is an object
    if not entity.endswith('_obj'):
        return False  # Only applicable to objects
    
    # Step 2: Derive the corresponding word entity (e.g., "baba_obj" -> "baba_word")
    base_name = entity[:-4]  # Remove '_obj' suffix
    word_entity = f"{base_name}_word"  # Corresponding word entity

    # Step 3: Check if the rule 'X_word is_word push_word' is formed
    if rule_formed(state, word_entity, "is_word", "push_word"):
        return True
    
    return False

# Automatically generated placeholders for missing predicates


def controllable(state, obj):
    """
    Check if an object is controllable.

    Args:
        state (dict): The game state.
        obj (str): The name of the object to check.

    Returns:
        bool: True if the object is controllable, False otherwise.
    """
    return obj == "red_agent"



def carrying(state, obj):
    """
    Check if the agent is carrying a specific object.

    Args:
        state (dict): The game state.
        obj (str): The name of the object to check (e.g., 'red_key').

    Returns:
        bool: True if the agent is carrying the specified object, False otherwise.
    """
    # Extract the type and color of the object
    if "_" not in obj:
        return False  # Invalid format; objects should have a "type_color" format
    
    obj_color, obj_type = obj.split("_")
    
    # Check if the agent is carrying the object
    for item in state.get("agent_carrying", []):
        if item["type"] == obj_type and item["color"] == obj_color:
            return True
    return False



def open_door(state, door):
    """
    Check if a door is open based on the state.

    Args:
        state (dict): The game state.
        door (str): The name of the door to check (e.g., 'red_door').

    Returns:
        bool: True if the door is open, False otherwise.
    """
    # Form the expected key for an open door
    open_door_key = f"open_{door}"
    # Check if the door key exists in the state
    return open_door_key in state



def locked(state, door):
    """
    Check if a door is locked based on the state.

    Args:
        state (dict): The game state.
        door (str): The name of the door to check (e.g., 'red_door').

    Returns:
        bool: True if the door is locked, False otherwise.
    """
    # Form the expected key for a locked door
    locked_door_key = f"locked_{door}"
    # Check if the locked door key exists in the state
    return locked_door_key in state



def next_to(state, obj1, obj2):
    """
    Check if two objects are adjacent (horizontally or vertically).

    Args:
        state (dict): The game state.
        obj1 (str): The name of the first object.
        obj2 (str): The name of the second object.

    Returns:
        bool: True if the objects are adjacent, False otherwise.
    """
    coords1 = state.get(obj1, [])
    coords2 = state.get(obj2, [])

    for (x1, y1) in coords1:
        for (x2, y2) in coords2:
            # Check if the objects are adjacent horizontally or vertically
            if abs(x1 - x2) + abs(y1 - y2) == 1:  # Manhattan distance = 1
                return True

    return False



def unlocks(state, key, door):
    """
    Check if a key can unlock a specific door based on their color.

    Args:
        state (dict): The game state.
        key (str): The name of the key (e.g., 'red_key').
        door (str): The name of the door (e.g., 'red_door').

    Returns:
        bool: True if the key unlocks the door, False otherwise.
    """
    # Extract the color of the key and the door
    key_color = key.split("_")[0]
    door_color = door.split("_")[0]

    # Check if the colors match
    return key_color == door_color



def blocking(state, obj, door):
    """
    Check if a specific object is blocking a door (i.e., adjacent to it).

    Args:
        state (dict): The game state.
        obj (str): The name of the object to check (e.g., 'red_ball').
        door (str): The name of the door to check (e.g., 'red_door').

    Returns:
        bool: True if the object is blocking the door, False otherwise.
    """
    # Combine all possible door representations
    possible_door_keys = [f"{prefix}_{door}" for prefix in ["locked", "closed", "open"]]
    door_coords = []
    for key in possible_door_keys:
        door_coords.extend(state.get(key, []))

    obj_coords = state.get(obj, [])

    # Check if any coordinate of the object is adjacent to the door
    for (dx, dy) in door_coords:
        for (ox, oy) in obj_coords:
            if abs(dx - ox) + abs(dy - oy) == 1:  # Manhattan distance = 1
                return True

    return False



def clear(state, door):
    """
    Check if a door is clear (i.e., no objects except excluded entities like 'grey_wall',
    'red_agent', 'agent_direction', and 'agent_carrying' are adjacent to it).

    Args:
        state (dict): The game state.
        door (str): The name of the door to check (e.g., 'red_door').

    Returns:
        bool: True if the door is clear, False otherwise.
    """
    # Combine all possible door representations
    possible_door_keys = [f"{prefix}_{door}" for prefix in ["locked", "closed", "open"]]
    door_coords = []
    for key in possible_door_keys:
        door_coords.extend(state.get(key, []))

    # Objects to exclude from "blocking" the door
    excluded_objects = {"grey_wall", "red_agent", "agent_direction", "agent_carrying"} | set(possible_door_keys)

    # Iterate over all objects in the state
    for obj, obj_coords in state.items():
        if obj in excluded_objects:  # Skip excluded objects
            continue

        # Check if any coordinate of the object is adjacent to the door
        for (dx, dy) in door_coords:
            for (ox, oy) in obj_coords:
                if abs(dx - ox) + abs(dy - oy) == 1:  # Manhattan distance = 1
                    return False

    return True






def inventory_full(state):
    """
    Check if the agent's inventory is full.

    Args:
        state (dict): The game state.

    Returns:
        bool: True if the inventory is full, False otherwise.
    """
    # Check if the agent is carrying "none"
    for item in state.get("agent_carrying", []):
        if item["type"] == "none" and item["color"] == "none":
            return False
    return True

