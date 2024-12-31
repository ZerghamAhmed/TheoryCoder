# from worldmodel2 import transition_model
from worldmodel import transition_model
from copy import deepcopy
from pathlib import Path
from predicates import *
from preprocessing import checker

def update_entity_categorizations(state):
    """
    Update the state dictionary with categorized entities like controllables, overlappables, pushables, and rules formed.

    Args:
        state (dict): The current game state.
    
    Returns:
        dict: The updated state with updated categorizations.
    """
    # Controllables are entities that match "entity_word is_word you_word"
    controllables = {
        entity for entity in state
        if rule_formed(state, f'{entity[:-4]}_word', 'is_word', 'you_word')  # Checks if entity follows rule: entity_word is you_word
    }
    
    # Overlappables are entities that are win_word or "goop_obj" or other specified objects
    overlappables = {
        entity for entity in state
        if rule_formed(state, f'{entity[:-4]}_word', 'is_word', 'win_word')  # Checks if entity follows rule: entity_word is win_word
        or entity.endswith('goop_obj')  # Example of overlappable goop
    }
    
    # # Pushables are typically word entities that can be pushed around
    # pushables = {
    #     entity for entity in state
    #     if entity.endswith('_word')  # Checks if the entity is a pushable word entity
    # }

    pushables = {
    entity for entity in state
        if entity.endswith('_word') 
        or rule_formed(state, f'{entity[:-4]}_word', 'is_word', 'push_word') 
        or (entity.endswith('_obj') and rule_formed(state, f'{entity[:-4]}_word', 'is_word', 'push_word'))
    }
    
    # Update state dictionary with these categorizations
    state['controllables'] = list(controllables)
    state['overlappables'] = list(overlappables)
    state['pushables'] = list(pushables)
    
    # Word entities on the map to track possible rules
    word_entities = [entity for entity in state.keys() if entity.endswith('_word')]
    rules_on_map = []
    
    # Iterate through word combinations to form potential rules
    for subj in word_entities:
        for pred in word_entities:
            for obj in word_entities:
                if rule_formed(state, subj, pred, obj):
                    print(f"Rule formed: {subj} {pred} {obj}")
                    rules_on_map.append(f'{subj} {pred} {obj}')
    
    # Update the 'rules_formed' key in the state dictionary with the formed rules
    state['rules_formed'] = rules_on_map

    if 'empty' in state:
        del state['empty']

    if 'won' in state:
        del state['won']
    
    return state


def convert_state_to_hashable(state):
    """
    Convert state dictionary to a hashable representation.
    """
    hashable_state = []
    for k, v in state.items():
        if isinstance(v, list):
            hashable_state.append((k, tuple(map(tuple, v))))
        elif isinstance(v, bool):
            hashable_state.append((k, v))
        elif isinstance(v, str):
            # Handle the string case, you can either keep it as a string or convert it into a tuple of characters
            hashable_state.append((k, v))  # Strings are hashable, so we can directly append them
        else:
            raise TypeError(f"Unsupported type in state: {type(v)} for key {k}")
    return frozenset(hashable_state)

# 50k was the baseline worked for mos 5000000
def enumerative_search(state0, operator, preconditions, effects, strategy='bfs', max_iters=500000, debug_callback=None, level=None):
    """
    Search for a goal state. Takes an optional `debug_callback` to call when errors occur.
    """
    from collections import deque

    actions_set = ["right", "up", "left", "down"]

    if rule_formed(state0, "keke_word", "is_word", "move_word"):
        actions_set.append("space")

    controllables = {
                    entity for entity in state0
                    if rule_formed(state0, f'{entity[:-4]}_word', 'is_word', 'you_word')
                }
    
    overlappables = {
                    entity for entity in state0
                    if rule_formed(state0, f'{entity[:-4]}_word', 'is_word', 'win_word') 
                    # or entity.endswith('goop_obj')

                }
    
    # pushables = {
    #                 entity for entity in state0
    #                 if entity.endswith('_word') or rule_formed(state0, f'{entity[:-4]}_word', 'is_word', 'push_word')
    #             }
    
    # pushables = {
    #                 entity for entity in state0
    #                 if entity.endswith('_word') or entity == 'rock_obj'
    #             }
    
    
    pushables = {
    entity for entity in state0
        if entity.endswith('_word') 
        or rule_formed(state0, f'{entity[:-4]}_word', 'is_word', 'push_word') 
        or (entity.endswith('_obj') and rule_formed(state0, f'{entity[:-4]}_word', 'is_word', 'push_word'))
    }


        
        # Add controllables to the state dictionary
    state0['controllables'] = list(controllables)
    state0['overlappables'] = list(overlappables)

    state0['pushables'] = list(pushables)

    word_entities = [entity for entity in state0.keys() if entity.endswith('_word')]
    rules_on_map = []
    for subj in word_entities:
        for pred in word_entities:
            for obj in word_entities:
                if rule_formed(state0, subj, pred, obj):
                    print(f"Rule formed: {subj} {pred} {obj}")
                    rules_on_map.append(subj + ' ' + pred + ' ' + obj)


    # Join the rules into a single string separated by commas (or another separator if needed)
    # state0['rules_formed'] = ', '.join(rules_on_map) if rules_on_map else ""
    state0['rules_formed'] = rules_on_map


    # breakpoint()


    if 'empty' in state0:
        del state0['empty']

    if 'won' in state0:
        del state0['won']

    start = ()
    states = {start: deepcopy(state0)}
    queue = deque([start])
    visited = set()
    search_iters = 0    

    while queue:
        # breakpoint()
        search_iters += 1
        if strategy == 'bfs':
            node = queue.popleft()
        else:
            node = queue.pop()

        current_state_hashable = convert_state_to_hashable(states[node])

        if current_state_hashable in visited:
            continue

        visited.add(current_state_hashable)

        if checker(states[node], effects, operator) and level != 2:
            break

        if search_iters > max_iters:
            print('MAX DEPTH REACHED')
            # breakpoint()
            return ["no-op"], states[node]

        for a in actions_set:
            try:
                # breakpoint()
                states[node] = update_entity_categorizations(states[node])
                # breakpoint()
                state = transition_model(deepcopy(states[node]), a)
                state = update_entity_categorizations(state)
                # if state.get('lost'):
                #     breakpoint()
                # Update the entity categorizations of the new state
                if state and state != states[node] and not state.get('lost'):
                    new_node = node + (a,)
                    states[new_node] = deepcopy(state)
                    queue.append(new_node)

                    # if rule_formed(state, "baba_word", "is_word", "you_word") and rule_formed(state, "baba_word", "is_word", "win_word"):
                    # if checker(state, effects, operator):
                    #     print('Goal reached')
                    #     # breakpoint()
                    #     return list(new_node), state
                    
                    # added exception for level 2 where PDDL plan effects cannot express dual condition
                    # that baba is you and baba is win must be satisfied simulataneously
                    if level == 2:
                        # breakpoint()
                        if (rule_formed(state, "baba_word", "is_word", "you_word") and rule_formed(state, "baba_word", "is_word", "win_word")):
                            print('Goal reached')
                            # breakpoint()
                            return list(new_node), state
                    else:
                        if checker(state, effects, operator):
                            print('Goal reached')
                            # breakpoint()
                            return list(new_node), state

            except Exception as e:  # Catch all exceptions
                print(f"Exception encountered: {e}. Triggering model debug.")
                if debug_callback:
                    debug_callback(states[node], a)  # Call the debug function with state and action
                else:
                    raise e  # Re-raise if no debug callback is provided
    # breakpoint()
    return list(node), states[node]

