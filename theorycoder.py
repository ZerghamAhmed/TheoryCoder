# theorycoder.py

import importlib
from pathlib import Path
from copy import deepcopy
import random
import inspect
import os

from agent_modules.llm_handler import QueryClient
from agent_modules.experiment_logger import ExperimentLogger
from agent_modules.modeling import WorldModelManager
from agent_modules.planning_module import HierarchicalPlanner
from agent_modules.experience_manager import ExperienceManager

import sys, os

# make sure we don't accidentally load /home/z/procgen/worldmodel.py
proj_root = os.path.abspath(os.path.dirname(__file__))
sys.path = [proj_root] 

from agent_modules.io_manager import (
    capture_world_model,
    extract_code_from_response,
    is_world_model_empty,
    load_plans,
    load_domain_pddl,
    load_predicates,
)


class TheoryCoderAgent:
    """
    Theory-based RL agent.

    Factorizes a world model into interaction rules and synthesizes code to predict
    next states. Can revise its model and replan hierarchically.
    """
    def __init__(
        self,
        world_model_load_name: str = "worldmodel",
        domain_file_name: str = "domain.pddl",
        predicates_file_name: str = "predicates",
        plans_file_name: str = "plans.json",
        json_reporter_path: str = None,
        language_model: str = "gpt-4o",
        query_mode: str = "openai",
        groq_model: str = "llama3-8b-8192",
        temperature: float = 0.7,
        episode_length: int = 20,
        do_revise_model: bool = False,
        planner_explore_prob: float = 0.0,
        max_replans: int = 1,
        base_dir: str = None,
        experiment_name: str = None,
        prune_plans: bool = False,
    ):
        # Config flags
        self.do_revise_model = do_revise_model
        self.planner_explore_prob = planner_explore_prob
        self.max_replans = max_replans
        self.prune_plans = prune_plans

        # File names
        self.world_model_load_name = world_model_load_name
        self.domain_file = domain_file_name
        self.predicates_file = predicates_file_name
        self.plans_file_name = plans_file_name

        # LLM client
        if query_mode == "openai":
            import openai
            self.llm_client = openai
        elif query_mode == "groq":
            from groq import Groq
            self.llm_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        else:
            raise ValueError(f"Unsupported query_mode: {query_mode}")

        self.query_lm = QueryClient(
            mode=query_mode,
            model=language_model,
            temperature=temperature,
            groq_model=groq_model,
        )

        # Experiment logger
        self.logger = ExperimentLogger(base_dir or os.getcwd(), experiment_name)

        # Runtime variables
        self.runtime_vars = {
            "observations": [],
            "exploratory_plans": [],
            "interaction_rules_str": {},
            "error_msg_model": "",
        }
        import utils
        self.runtime_vars["utils"] = inspect.getsource(utils)

        # Load static files
        self.runtime_vars["domain_file"] = load_domain_pddl(self.domain_file)
        self.runtime_vars["predicates"] = load_predicates(self.predicates_file)

        # Plans & experience buffer
        self.plans = load_plans(self.plans_file_name)
        self.replay_buffers = []
        self.experience_manager = ExperienceManager(
            replay_buffers=self.replay_buffers,
            call_model_debug=self.call_model_debug,
            get_pred_errors=lambda actual, pred: "" if actual == pred else f"expected {actual} but got {pred}",
        )

    def call_model_debug(self, state, action):
        """Reload worldmodel.py and call its transition_model()."""
        import worldmodel
        importlib.reload(worldmodel)
        try:
            return worldmodel.transition_model(state, action)
        except Exception:
            return state

    def _sample_planner_mode(self) -> str:
        return random.choices(
            ["exploit", "explore"],
            weights=[1 - self.planner_explore_prob, self.planner_explore_prob],
        )[0]

    def overwrite_world_model(self, new_code: str):
        """Persist the new worldmodel.py contents."""
        Path(f"{self.world_model_load_name}.py").write_text(new_code)

    def reset(self):
        """Reset the environment, reload worldmodel, and clear buffers."""
        self.engine.reset()
        # reload current worldmodel code into runtime_vars
        wm_text = capture_world_model(f"{self.world_model_load_name}.py")
        self.runtime_vars["world_model_str"] = wm_text
        # clear experience buffer & observations
        self.replay_buffers.clear()
        obs0 = deepcopy(self.engine.get_obs())
        self.runtime_vars["observations"] = [obs0]
        self.actions = []

    def execute_random_actions(self, num_actions: int):
        """Gather random transitions for initialization."""
        for _ in range(num_actions):
            a = random.choice(self.engine.actions_set)
            self.step_env(a)

    def step_env(self, action):
        """Apply action, record transition, append action to tape."""
        s0 = deepcopy(self.runtime_vars["observations"][-1])
        s1, reward, done, info = self.engine.step(action)
        s1 = deepcopy(s1)
        self.runtime_vars["observations"].append(s1)
        self.replay_buffers.append((s0, action, s1))
        self.tape[-1].setdefault("actions", []).append(action)

    def run(self, engine, max_revisions: int = 5, max_attempts: int = 6) -> bool:
        """
        Main loop:
         1) reset
         2) if worldmodel file empty → random explore + initialize
         3) else hierarchical plan
         4) execute plan, on loss revise & replan
        """
        self.engine = engine
        self.tape = [{}]
        revision_count = 0
        attempt_count = 0

        # Build managers now that engine exists
        self.wm_manager = WorldModelManager(
            runtime_vars=self.runtime_vars,
            engine=self.engine,
            logger=self.logger,
            query_lm=self.query_lm.query,            # use the .query() method
            extract_code=extract_code_from_response,
            overwrite_world_model=self.overwrite_world_model,
        )
        self.planner = HierarchicalPlanner(
            domain_file=self.domain_file,
            max_replans=self.max_replans,
        )

        while revision_count <= max_revisions and attempt_count < max_attempts:
            # 1) reset
            self.reset()
            first_letters = ""
            did_revise = False

            # 2) if world-model file empty, random explore + initialize
            wm_path = f"{self.world_model_load_name}.py"
            if is_world_model_empty(wm_path) and self.do_revise_model:
                print("Empty world model → random exploration + init")
                self.execute_random_actions(15)
                self.wm_manager.initialize_world_model(
                    num_actions=15,
                    choose_synthesis_examples=self.experience_manager.choose_synthesis_examples,
                )
                revision_count += 1
                continue

            # 3) hierarchical plan
            import worldmodel
            importlib.reload(worldmodel)
            importlib.invalidate_caches()
            importlib.reload(worldmodel)
            print("Using worldmodel.py from", worldmodel.__file__)
            breakpoint()
            mode = self._sample_planner_mode()
            plan = self.planner.plan(
                mode,
                state=self.runtime_vars["observations"][-1],
                plans=self.plans,
                debug_callback=self.wm_manager.debug_model,
                subplan_exploratory=None,
                engine=self.engine,
            )

            # 4) execute
            for a in plan:
                self.step_env(a)
                first_letters += a[0]
                if self.engine.won:
                    print("Agent won!")
                    return True
                if self.engine.lost:
                    print("Agent died, revising model")
                    self.wm_manager.revise_world_model(
                        choose_synthesis_examples=self.experience_manager.choose_synthesis_examples,
                        do_revise_model=self.do_revise_model,
                        do_revise_plan=lambda err: err > 0,
                    )
                    attempt_count += 1
                    revision_count += 1
                    did_revise = True
                    break

            # 5) if we revised, loop to replan
            if did_revise:
                continue

            # neither win nor further revision → exit
            break

        print("Run terminated without win.")
        return False
