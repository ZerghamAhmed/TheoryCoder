import importlib
from levelrunner import actor
import utils
from games import BabaIsYou
import random
from typing import Callable, List, Optional
import re
import os
from collections import deque
from predicates import *
from preprocessing import operator_extractor, checker
from copy import deepcopy
from agent_modules.planner import enumerative_search

def actor(domain_file, subplan, state, max_iterations=None, debug_callback=None, level=None, engine=None):
    operator = operator_extractor(domain_file, subplan)
    print("OPERATOR AND PRECONDS LIST", operator)

    preconditions = operator['preconditions']
    effects = operator['effects']

    actions, new_state = enumerative_search(
        state, operator, preconditions, effects, 
        debug_callback=debug_callback, level=level, engine=engine
    )
    if actions:
        return actions, new_state

class HierarchicalPlanner:
    """
    A standalone planner module that encapsulates hierarchical planning logic.
    """
    def __init__(self, domain_file: str, max_replans: int = 1):
        self.domain_file = domain_file
        self.max_replans = max_replans

    def choose_mode(self, explore_prob: float) -> str:
        """
        Decide whether to "explore" or "exploit" based on a probability.
        """
        if random.random() < explore_prob:
            return 'explore'
        return 'exploit'

    def plan(self,
             mode: str,
             state: dict,
             plans: dict,
             debug_callback: Callable,
             subplan_exploratory: Optional[str] = None,
             engine=None,
             max_iterations=None) -> List[str]:
        """
        Entry point: returns a flat list of actions given mode.
        """
        if mode == 'explore':
            return self.plan_explore(state, subplan_exploratory, debug_callback, engine)
        else:
            return self.plan_exploit(state, plans, debug_callback, engine)

    def plan_explore(self,
                      state: dict,
                      subplan: str,
                      debug_callback: Callable,
                      engine) -> List[str]:
        """
        Execute a single exploratory subplan.
        """
        # reload world model & planner dependencies
        import worldmodel, agent_modules.planner as planner, levelrunner
        importlib.reload(worldmodel)
        importlib.reload(planner)
        importlib.reload(levelrunner)

        actions, _ = actor(
            self.domain_file,
            subplan,
            state,
            max_iterations=None,
            debug_callback=debug_callback,
            level=None,
            engine=engine
        )
        return actions

    def plan_exploit(self,
                      state: dict,
                      plans: dict,
                      debug_callback: Callable,
                      engine) -> List[str]:
        """
        Loop through each high‐level subplan and synthesize low‐level actions.
        """
        actions = []
        # reload deps each iteration
        for i in range(self.max_replans):
            import worldmodel, agent_modules.planner as planner, levelrunner
            importlib.reload(worldmodel)
            importlib.reload(planner)
            importlib.reload(levelrunner)
            importlib.reload(utils)

            for subplan in plans.get(str(engine.level_id), []):
                seq, state = actor(
                    self.domain_file,
                    subplan,
                    state,
                    max_iterations=None,
                    debug_callback=debug_callback,
                    level=engine.level_id,
                    engine=engine
                )
                actions.extend(seq)
        return actions
