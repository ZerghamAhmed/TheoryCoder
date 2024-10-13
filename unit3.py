# Assuming all the predicates above are defined 

# Import necessary modules
from itertools import product
from games import BabaIsYou
from predicates import *

# Initialize your engine and define the levels to analyze
levels = range(0, 14)  # Levels 0 to 13

for level in levels:
    # Initialize the engine for the current level
    engine = BabaIsYou(level_set='demo_LEVELS', level_id=level)
    
    # Get the current state
    state = engine.get_obs()
    
    # Convert all tuple positions to lists for consistency
    state = {key: [list(item) for item in value] if isinstance(value, list) else value for key, value in state.items()}
    
    # Identify controllable entities based on the "is you" rule
    controllables = {
        entity for entity in state
        if rule_formed(state, f'{entity[:-4]}_word', 'is_word', 'you_word')
    }
    
    # Add controllables to the state dictionary
    state['controllables'] = list(controllables)
    
    # Extract all word entities (entities ending with '_word')
    word_entities = [entity for entity in state if entity.endswith('_word')]
    
    # Define possible rules: (subject_word, 'is_word', predicate_word)
    possible_rules = []
    for subject in word_entities:
        for predicate in word_entities:
            possible_rules.append( (subject, 'is_word', predicate) )
    
    # Initialize lists to hold breakable and formable rules
    breakable_rules = []
    formable_rules = []
    
    # Check each possible rule for breakability and formability
    for rule in possible_rules:
        if rule_breakable(state, *rule):
            breakable_rules.append(rule)
        
        if rule_formable(state, *rule):
            formable_rules.append(rule)
    
    # Initialize a dictionary to hold pushable directions for each word
    pushable_words = {}
    for word in word_entities:
        pushable_words[word] = {
            'up': pushable_word_up(state, word),
            'down': pushable_word_down(state, word),
            'left': pushable_word_left(state, word),
            'right': pushable_word_right(state, word)
        }
    
    # Collect all formed rule coordinates to exclude them from the formable rules
    formed_rule_coordinates = set()
    for word1, word2, word3 in possible_rules:
        if not rule_breakable(state, word1, word2, word3):
            for triplet in product(state.get(word1, []), state.get(word2, []), state.get(word3, [])):
                if are_adjacent(list(triplet)):
                    formed_rule_coordinates.update(tuple(coord) for coord in triplet)

    # Print the results for the current level
    print(f"Level {level}:")
    
    # Print Breakable Rules
    print("  Breakable Rules:")
    if breakable_rules:
        for rule in breakable_rules:
            rule_str = f"{rule[0].replace('_word', '').capitalize()} is {rule[2].replace('_word', '').capitalize()}"
            print(f"    {rule}: Breakable")
    else:
        print("    None")
    
    # Print Formable Rules with Coordinates and Indices
    print("  Formable Rules:")
    if formable_rules:
        for rule in formable_rules:
            rule_str = f"{rule[0].replace('_word', '').capitalize()} is {rule[2].replace('_word', '').capitalize()}"
            print(f"    {rule}: Formable")
            
            # Extract coordinates for each word in the rule
            word1, word2, word3 = rule
            c1 = state.get(word1, [])
            c2 = state.get(word2, [])
            c3 = state.get(word3, [])
            instant = []
            
            # Collect all possible triplets for the current formable rule
            for triplet in product(c1, c2, c3):
                instant.append(list(triplet))
            
            # Exclude triplets that have any coordinate already part of a formed rule
            valid_triplets = []
            for triplet in instant:
                if all(tuple(coord) not in formed_rule_coordinates for coord in triplet):
                    valid_triplets.append(triplet)
            
            # Print the valid triplets with their coordinates and indices
            if valid_triplets:
                for triplet in valid_triplets:
                    coord1, coord2, coord3 = triplet
                    # Find the indices of each coordinate in their respective lists
                    try:
                        idx1 = c1.index(coord1)
                    except ValueError:
                        idx1 = -1  # Coordinate not found
                    try:
                        idx2 = c2.index(coord2)
                    except ValueError:
                        idx2 = -1  # Coordinate not found
                    try:
                        idx3 = c3.index(coord3)
                    except ValueError:
                        idx3 = -1  # Coordinate not found
                    
                    print(f"      Coordinates: {triplet}, Indices: [{idx1}, {idx2}, {idx3}]")
            else:
                print(f"      No valid triplet found for this rule.")
    else:
        print("    None")
    
    print("\n" + "-"*50 + "\n")
