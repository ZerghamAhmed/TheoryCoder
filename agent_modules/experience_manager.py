from typing import Any, List, Tuple, Optional, Callable

# A transition tuple: (previous_state, action, next_state)
Transition = Tuple[Any, Any, Any]

class ExperienceManager:
    """
    Encapsulates the replay buffer and logic for generating
    program-synthesis examples from past transitions.
    """
    def __init__(
        self,
        replay_buffers: List[Transition],
        call_model_debug: Callable[[Any, Any], Any],
        get_pred_errors: Callable[[Any, Any], str]
    ):
        """
        :param replay_buffers: list of (state0, action, state1) tuples
        :param call_model_debug: function(state0, action) -> predicted state1
        :param get_pred_errors: function(actual state1, predicted state1) -> error string
        """
        self.replay_buffers = replay_buffers
        self._call_model_debug = call_model_debug
        self._get_pred_errors = get_pred_errors

    def make_observation_summaries(self, obs: Transition, errors: str) -> str:
        """
        Format a single transition and its prediction errors into a string.
        """
        s0, a, s1 = obs
        return (
            f"Initial state: {s0}\n"
            f"Action: {a}\n"
            f"Next state: {s1}\n"
            f"\nYour prediction errors:\n{errors}\n"
        )

    def choose_synthesis_examples(
        self,
        exploratory_plan: Optional[str] = None
    ) -> Tuple[List[str], int]:
        """
        Choose (s0, a) -> s1 transitions from the replay buffer as program synthesis examples.

        :param exploratory_plan: label for grouping examples (optional)
        :return: (formatted_examples, error_count)
        """
        # 1. Copy buffer
        obs = self.replay_buffers[::1]

        # 2. Generate predictions
        preds = [
            self._call_model_debug(s0, a)
            for (s0, a, s1) in obs
        ]

        # 3. Compute errors
        errors = [
            self._get_pred_errors(s1, pred)
            for (s0, a, s1), pred in zip(obs, preds)
        ]

        # 4. Create summaries
        examples = [
            self.make_observation_summaries((s0, a, s1), e)
            for (s0, a, s1), e in zip(obs, errors)
        ]

        # 5. Count non-empty errors
        error_count = sum(1 for e in errors if e)

        # 6. Optionally prefix with exploratory plan header
        if exploratory_plan:
            formatted_examples = [
                f"ERRORS FROM WORLD MODEL for EXPLORATORY PLAN {exploratory_plan}:\n\n" +
                "\n\n".join(examples)
            ]
        else:
            formatted_examples = examples

        return formatted_examples, error_count

    def sample_replay_buffer(self, k: int) -> List[Transition]:
        """Randomly sample k transitions from the buffer."""
        import random
        return random.sample(self.replay_buffers, k)

    def __len__(self) -> int:
        """Current number of stored transitions."""
        return len(self.replay_buffers)
