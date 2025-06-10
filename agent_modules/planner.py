# from worldmodel2 import transition_model
# from worldmodel import *
from worldmodel import *
from copy import deepcopy
from pathlib import Path
from predicates import *
from preprocessing import checker



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






import time
# 50k was the baseline worked for mos 20k worked for oracle model
# 90000000000000
def enumerative_search(state0, operator, preconditions, effects, strategy='bfs', max_iters=100000, debug_callback=None, level=None, engine=None):
   """
   Search for a goal state. Takes an optional `debug_callback` to call when errors occur.
   """
   from collections import deque


   start_time = time.time()

   # try reloading predicates
#    import importlib
#    importlib.reload(predicates)

#    breakpoint()
   # actions_set = ["noop", "right", "left", "up", "down", "swing"]
   actions_set = ["right", "left", "up", "down"]
   # actions_set = ["up", "down", "right", "left"]
   # actions_set = ["right", "down"]
   # actions_set = ["right", "down", "up"]






#    breakpoint()






   start = []
   states = {tuple(start): deepcopy(state0)}
   queue = deque([start])
   visited = set()
   search_iters = 0   
#    breakpoint()


   while queue:
       # breakpoint()
       search_iters += 1
       if strategy == 'bfs':
           node = queue.popleft()
       else:
           node = queue.pop()


    #    current_state_hashable = convert_state_to_hashable(states[tuple(node)])


    #    if current_state_hashable in visited:
    #        continue


    #    visited.add(current_state_hashable)


       if checker(states[tuple(node)], effects, operator):
           # breakpoint()
           break


       if search_iters > max_iters:
           print('MAX DEPTH REACHED')
           # breakpoint()
           elapsed_time = time.time() - start_time
           print(f"Planner terminated after {elapsed_time:.2f} seconds")
           return list(new_node), states[tuple(node)]

    #    breakpoint()
       for a in actions_set:
           try:
               # breakpoint()
               import importlib
               import worldmodel
               importlib.reload(worldmodel)
               states[tuple(node)] = update_entity_categorizations(states[tuple(node)])
            #    breakpoint()
               state = transition_model(deepcopy(states[tuple(node)]), a)
               state = update_entity_categorizations(state)
              
               if state and state != states[tuple(node)] and state['avatar'] != []:
                   new_node = node + [a]
                   states[tuple(new_node)] = deepcopy(state)
                   queue.append(new_node)


                   state_hashable = convert_state_to_hashable(state)


                   if checker(state, effects, operator):
                       print('Goal reached')
                       elapsed_time = time.time() - start_time
                       print(f"Goal reached in {elapsed_time:.2f} seconds")
                       print(search_iters)
                       # breakpoint()


                       # for action in list(new_node):
                       #     engine.step_env(action)
                       #     engine.save_screen()


                       return list(new_node), state


           except Exception as e:  # Catch all exceptions
            #    breakpoint()
               print(f"Exception encountered: {e}. Triggering model debug.")
            #    breakpoint()
               if debug_callback:
                   debug_callback(states[tuple(node)], a)  # Call the debug function with state and action
               else:
                   raise e  # Re-raise if no debug callback is provided
   # breakpoint()
   elapsed_time = time.time() - start_time
   print(f"Planner terminated after {elapsed_time:.2f} seconds without finding a goal.")
   return list(node), states[tuple(node)]











