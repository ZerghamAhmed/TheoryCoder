# from worldmodel2 import transition_model
from copy import deepcopy
from pathlib import Path
import importlib.util
import os
from preprocessing import checker
import preprocessing

# Location of the world model is configured via environment variable
WORLDMODEL_FILE = os.getenv("TC_WORLDMODEL_FILE", "worldmodel.py")

def update_entity_categorizations(state):
    """
    Update the state dictionary with latent variables.

    Args:
        state (dict): The current game state.
    
    Returns:
        dict: The updated state with updated categorizations.
    """
    
    return state

def convert_state_to_hashable(state):
    """
    Recursively convert a state into a hashable form.

    Args:
        state (dict or list): The state structure to convert.

    Returns:
        hashable: A hashable representation of the state.
    """
    if isinstance(state, dict):
        # Convert dictionary to a tuple of sorted key-value pairs
        return tuple((key, convert_state_to_hashable(value)) for key, value in sorted(state.items()))
    elif isinstance(state, list):
        # Convert lists to tuples
        return tuple(convert_state_to_hashable(item) for item in state)
    elif isinstance(state, tuple):
        # Recursively handle tuples
        return tuple(convert_state_to_hashable(item) for item in state)
    else:
        # Return immutable types as is
        return state




# 50k was the baseline worked for mos 20k worked for oracle model
def enumerative_search(state0, operator, preconditions, effects, strategy='bfs', max_iters=9000000,
                       debug_callback=None, predicate_debug_callback=None, level=None, engine=None,
                       timeout=None):
    """
    Search for a goal state. Takes an optional `debug_callback` to call when errors occur.

    Args:
        timeout: Maximum time in seconds for the search. None means no timeout.
    """
    from collections import deque
    import time

    start_time = time.time() if timeout else None

    # Dynamically load the world model so it can reside outside the CWD
    spec = importlib.util.spec_from_file_location("worldmodel", WORLDMODEL_FILE)
    worldmodel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worldmodel)
    transition_model = worldmodel.transition_model

    # Determine the available actions. If an engine/environment with an
    # `actions_set` attribute is provided, use it. Otherwise fall back to a
    # default minimal action set.
    if engine is not None and hasattr(engine, "actions_set"):
        actions_set = list(engine.actions_set)
    else:
        # actions_set = ["noop", "right", "left", "up", "down", "swing"]
        actions_set = ["right", "left", "up", "down", "noop"]


    # breakpoint()



    start = []
    states = {tuple(start): deepcopy(state0)}
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

        current_state_hashable = convert_state_to_hashable(states[tuple(node)])

        if current_state_hashable in visited:
            continue

        visited.add(current_state_hashable)

        try:
            if checker(states[tuple(node)], effects, operator):
                break
        except preprocessing.PredicateExecutionError as e:
            print(f"Predicate error encountered: {e}. Triggering predicate debug.")
            if predicate_debug_callback:
                predicate_debug_callback(states[tuple(node)], operator, e)
                if checker(states[tuple(node)], effects, operator):
                    break
            else:
                raise e

        if search_iters > max_iters:
            print('MAX DEPTH REACHED')
            # breakpoint()
            return list(new_node), states[tuple(node)]

        # Check timeout
        if timeout and (time.time() - start_time) > timeout:
            print(f'PLANNER TIMEOUT ({timeout}s)')
            return list(node), states[tuple(node)]

        for a in actions_set:
            try:
                # breakpoint()
                states[tuple(node)] = update_entity_categorizations(states[tuple(node)])
                # breakpoint()
                state = transition_model(deepcopy(states[tuple(node)]), a)

                state = update_entity_categorizations(state)
                
                if state and state != states[tuple(node)]:
                    new_node = node + [a]
                    states[tuple(new_node)] = deepcopy(state)
                    queue.append(new_node)

                    try:
                        if checker(state, effects, operator):
                            print('Goal reached')
                            return list(new_node), state
                    except preprocessing.PredicateExecutionError as e:
                        print(f"Predicate error encountered: {e}. Triggering predicate debug.")
                        if predicate_debug_callback:
                            predicate_debug_callback(state, operator, e)
                            if checker(state, effects, operator):
                                print('Goal reached')
                                return list(new_node), state
                        else:
                            raise e

            except Exception as e:  # Catch all exceptions
                print(f"Exception encountered: {e}. Triggering model debug.")
                if debug_callback:
                    debug_callback(states[tuple(node)], a)  # Call the debug function with state and action
                    # Return empty action list so caller can handle failure
                    return [], states[tuple(node)]
                else:
                    raise e  # Re-raise if no debug callback is provided
    
    return list(node), states[tuple(node)]

