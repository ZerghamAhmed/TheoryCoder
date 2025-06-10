import os
import importlib
from typing import Callable, Any, Tuple, List

from prompts import (
    initialize_world_model_prompt,
    revise_world_model_prompt,
    debug_model_prompt,
)

class WorldModelManager:
    def __init__(
        self,
        runtime_vars: dict,
        engine,
        logger,
        query_lm: Callable[[str], Tuple[str, Any]],
        extract_code: Callable[[str], str],
        overwrite_world_model: Callable[[str], None],
    ):
        self.runtime_vars = runtime_vars
        self.engine = engine
        self.logger = logger
        self.query_lm = query_lm
        self.extract_code = extract_code
        self.overwrite_world_model = overwrite_world_model

        # you could also pull your prompts in here
        from prompts import (
            initialize_world_model_prompt,
            revise_world_model_prompt,
            debug_model_prompt,
        )
        self.init_prompt = initialize_world_model_prompt
        self.revise_prompt = revise_world_model_prompt
        self.debug_prompt = debug_model_prompt

    def initialize_world_model(self, num_actions: int, choose_synthesis_examples: Callable[[], Tuple[List[str], int]]):
        examples, error_count = choose_synthesis_examples()
        prompt = self.init_prompt.format(
            current_state=self.runtime_vars['observations'][-1],
            actions_set=self.engine.actions_set,
            num_random_actions=num_actions,
            errors_from_world_model='\n\n'.join(examples),
            utils="directions = {{ 'left': [-1, 0], 'right': [1, 0], 'up': [0, 1], 'down': [0, -1] }}"
        )
        print(prompt)
        with open('current_prompt_INIT.txt','w') as f:
            f.write(prompt)
        resp, fingerprint = self.query_lm(prompt)
        code = self.extract_code(resp)
        if code:
            self.runtime_vars['world_model_str'] = code
            self.overwrite_world_model(code)

            step = self.logger.create_step("initialize")
            self.logger.save_step_files(step, prompt, resp, code, "worldmodel.py")
            with open(os.path.join(step, "fingerprint.txt"), "w") as f:
                f.write(str(fingerprint))
            self.logger.add_to_tape({
                "step":"initialize", "prompt": prompt, "response": resp
            })

    def revise_world_model(self,
               choose_synthesis_examples: Callable[[], Tuple[List[str], int]],
               do_revise_model: bool,
               do_revise_plan: Callable[[int], bool]):
        if not do_revise_model:
            return

        self.logger.tape[-1].update({'revision_prompts': {}, 'revision_responses': {}})
        examples, err = choose_synthesis_examples()
        if err <= 0:
            return

        prompt = self.revise_prompt.format(
            actions_set=self.engine.actions_set,
            errors_from_world_model='\n\n'.join(examples),
            world_model_str=self.runtime_vars['world_model_str'],
            utils="directions = {{ 'left': [-1,0], 'right': [1,0], 'up':[0,1], 'down':[0,-1] }}"
        )
        print(prompt)
        step = self.logger.create_step("revision")
        resp, fingerprint = self.query_lm(prompt)
        code = self.extract_code(resp)
        if code:
            self.logger.save_step_files(step, prompt, resp, code, "worldmodel.py")
            with open(os.path.join(step, "fingerprint.txt"), "w") as f:
                f.write(str(fingerprint))
            self.logger.add_to_tape({"step":"revision","prompt":prompt,"response":resp})

            # snapshot plans, update world‐model
            with open(os.path.join(step, "pruned_plans.txt"), "w") as f:
                f.write('\n'.join(self.runtime_vars.get('exploratory_plans', [])))
            self.runtime_vars['world_model_str'] = code
            self.runtime_vars['error_msg_model'] = code
            self.overwrite_world_model(code)

        # set flag if we should replan
        if do_revise_plan(err):
            self.runtime_vars['revise_plan'] = True

    def debug_model(self, state, action, max_retries=3):
        for i in range(max_retries):
            try:
                import worldmodel; import importlib
                importlib.reload(worldmodel)
                return worldmodel.transition_model(state, action)
            except Exception as e:
                print(f"DEBUG ITER {i} error: {e}")
                prompt = self.debug_prompt.format(
                    actions_set=self.engine.actions_set,
                    world_model_str=self.runtime_vars['world_model_str'],
                    observations='IGNORE',
                    state=state,
                    action=action,
                    error=e,
                    utils="directions = {{ 'left': [-1,0], 'right': [1,0], 'up':[0,1], 'down':[0,-1] }}"
                )
                step = self.logger.create_step("debug")
                resp, fingerprint = self.query_lm(prompt)
                code = self.extract_code(resp)
                if code:
                    self.logger.save_step_files(step, prompt, resp, code, "worldmodel.py")
                    with open(os.path.join(step, "fingerprint.txt"), "w") as f:
                        f.write(str(fingerprint))
                    self.logger.add_to_tape({
                        "step":"debug","prompt":prompt,"response":resp,"error":str(e)
                    })
                    self.runtime_vars['world_model_str'] = code
                    self.runtime_vars['error_msg_model'] = code
                    self.overwrite_world_model(code)
        return None
