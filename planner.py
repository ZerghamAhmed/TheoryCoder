# from worldmodel2 import transition_model
from worldmodel import transition_model
from copy import deepcopy
from pathlib import Path
from predicates import *
from preprocessing import checker

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
        else:
            raise TypeError(f"Unsupported type in state: {type(v)} for key {k}")
    return frozenset(hashable_state)

# 50k was the baseline worked for mos
def enumerative_search(state0, operator, preconditions, effects, strategy='bfs', max_iters=500000, debug_callback=None):
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
        
        # Add controllables to the state dictionary
    state0['controllables'] = list(controllables)

    start = ()
    states = {start: deepcopy(state0)}
    queue = deque([start])
    visited = set()
    search_iters = 0    

    while queue:
        search_iters += 1
        if strategy == 'bfs':
            node = queue.popleft()
        else:
            node = queue.pop()

        current_state_hashable = convert_state_to_hashable(states[node])

        if current_state_hashable in visited:
            continue

        visited.add(current_state_hashable)

        if checker(states[node], effects, operator):
            break

        if search_iters > max_iters:
            print('MAX DEPTH REACHED')
            # breakpoint()
            return ["no-op"], states[node]

        for a in actions_set:
            try:
                state = transition_model(deepcopy(states[node]), a)
                if state and state != states[node] and not state.get('lost'):
                    new_node = node + (a,)
                    states[new_node] = deepcopy(state)
                    queue.append(new_node)

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

