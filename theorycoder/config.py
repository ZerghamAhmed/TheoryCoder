"""Configuration dataclass for TheoryCoderAgent."""
from dataclasses import dataclass, field
from typing import Optional, Set, Dict, Union


@dataclass
class AgentConfig:
    """Configuration for TheoryCoderAgent.

    This dataclass encapsulates all configuration parameters for the agent,
    making it easier to understand and modify the agent's behavior.
    """
    # File loading
    world_model_load_name: Optional[str] = None
    operators_load_name: Optional[str] = None
    predicates_load_name: Optional[str] = None
    json_reporter_path: Optional[str] = None

    # LLM settings
    language_model: str = 'gpt-4o-2024-11-20'
    query_mode: str = 'openai_direct'  # 'langchain_openai', 'openai_direct', 'groq', 'custom'
    groq_model: str = "llama3-8b-8192"
    reasoning_effort: Optional[str] = "high"  # "low" | "medium" | "high" | None
    use_responses_api: bool = False
    temperature: float = 1.0

    # PDDL settings
    domain_file_name: str = 'domain.pddl'
    predicates_file_name: str = 'predicates.py'
    plans_file_name: str = 'plans.json'
    fast_downward_path: Optional[str] = None

    # Experiment settings
    base_dir: Optional[str] = None
    experiment_name: Optional[str] = None
    create_subdir: bool = True
    centralize_files: bool = True
    use_custom_pddl_prompts: bool = False
    replace_predicates: bool = False

    # Agent behavior
    episode_length: int = 20
    do_revise_model: bool = False
    sparse_interactions: bool = True
    observation_memory_size: int = 1
    planner_explore_prob: float = 0.0
    max_replans: int = 1
    prune_plans: bool = False

    # Transfer settings
    transfer_levels: Union[Set[int], Dict[str, Set[int]], None] = None

    # Revision settings
    revision_mode: str = "hybrid"  # "full", "errors_only", "limited", "hybrid"
    max_revision_transitions: int = 50
    max_revision_tokens: int = 30000

    def __post_init__(self):
        """Normalize transfer_levels to a consistent format."""
        if self.transfer_levels is None:
            self.transfer_levels = set()


@dataclass
class RuntimeState:
    """Runtime state that changes during agent execution.

    Separates mutable runtime state from configuration for clarity.
    """
    interaction_rules: Dict = field(default_factory=dict)
    interaction_rules_str: Dict = field(default_factory=dict)
    error_msg_model: str = ''
    observations: list = field(default_factory=list)
    revise_plan: bool = False
    plan_str: str = ''
    plan_log: str = ''
    goal: str = 'Win'
    goal_state_str: str = ''
    operators: str = ''
    predicates: str = ''
    worldmodel: str = ''
    observed_collisions: str = ''
    unobserved_collisions: str = ''
    previous_entities_encountered: list = field(default_factory=list)
    new_entities_encountered: list = field(default_factory=list)
    exploratory_plans: list = field(default_factory=list)
    unsatisfied_preconditions: list = field(default_factory=list)
    world_model_str: str = ''
    utils: str = ''
