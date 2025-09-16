import re
import os
from collections import deque
from predicates import *
from preprocessing import operator_extractor, checker
from copy import deepcopy
from planner import enumerative_search
# import planner


def actor(domain_file, subplan, state, max_iterations=None, debug_callback=None, level=None):
    operator = operator_extractor(domain_file, subplan)
    print("OPERATOR AND PRECONDS LIST", operator)


    preconditions = operator['preconditions']
    effects = operator['effects']


    actions, new_state = enumerative_search(
        state, operator, preconditions, effects,
        debug_callback=debug_callback, level=level,
    )
    if actions:
        return actions, new_state




    # # Evaluate preconditions
    # precondition_results = checker(state, preconditions, operator)
    # if not precondition_results:
    #     unsatisfied_preconds = [
    #         pre for pre in operator["preconditions"]
    #         if not checker(state, [pre], operator)
    #     ]
    #     error_message = (
    #         f"Preconditions not satisfied to execute {subplan}. "
    #         f"Unsatisfied preconditions: {unsatisfied_preconds}"
    #     )
    #     print(error_message)
    #     breakpoint()
    #     raise ValueError(error_message)
    # else:
    #     print(f"Preconditions satisfied to execute {subplan}")
    #     # breakpoint()
    #     actions, new_state = enumerative_search(
    #         state, operator, preconditions, effects,
    #         debug_callback=debug_callback, level=level
    #     )
    #     if actions:
    #         return actions, new_state
    #     # else:
    #         # breakpoint()







