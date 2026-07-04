import re
import os
from collections import deque
from preprocessing import operator_extractor, checker, set_domain_file
from copy import deepcopy
from planner import enumerative_search
import planner

def actor(domain_file, subplan, state, max_iterations=None, debug_callback=None,
          predicate_debug_callback=None, level=None, engine=None, timeout=None):
    # Ensure predicate mappings correspond to the provided domain file
    set_domain_file(domain_file)

    operator = operator_extractor(domain_file, subplan)
    print("OPERATOR AND PRECONDS LIST", operator)

    preconditions = operator['preconditions']
    effects = operator['effects']


    result = enumerative_search(
        state, operator, preconditions, effects,
        debug_callback=debug_callback,
        predicate_debug_callback=predicate_debug_callback,
        level=level,
        engine=engine,
        timeout=timeout
    )

    if result:
        actions, new_state = result
    else:
        actions, new_state = [], state

    if actions:
        return actions, new_state

    # If no actions were found, return an empty list and current state
    return [], new_state

