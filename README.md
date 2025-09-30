# TheoryCoder

This repository contains tooling for automatically learning world models for
grid-based games and transferring those models between games.

## Generating a World Model from Scratch

`theorycoder3.py` is the main entry point for learning a model. The `--scratch`
flag removes any existing generated files so the agent starts fresh. Use
`--learn-model` to allow the agent to update its world model after each run.

Example command for learning the Labyrinth game from scratch:

```bash
python theorycoder3.py \
  --game labyrinth \
  --level-sets "{'labyrinth': [0]}" \
  --episode-length 20 \
  --experiment-dir my_experiment \
  --scratch \
  --learn-model
```

Running this command creates a new directory under `experiments/my_experiment`
containing the world model (`worldmodel.py`), predicates, domain/problem PDDL
files, and a run tape.

## Transferring a Learned Model

`transfer_runner.py` copies a previously learned model from one game and applies
it to others. Specify the experiment directory and a transfer specification of
the form `src:dst1,dst2,...`.

Example command transferring a Labyrinth model to Sokoban and Maze:

```bash
python transfer_runner.py \
  --experiment-dir my_experiment \
  --transfer labyrinth:sokoban,maze \
  --learn-model
```

This command finds the latest run under `experiments/my_experiment`, copies the
Labyrinth world model into `tc_game/sokoban` and `tc_game/maze`, and then runs
`TheoryCoderAgent` on each destination environment. When `--learn-model` is
passed the runner first attempts to solve each destination using the transferred
model unchanged. If that attempt fails, the agent automatically initializes a
new world model for the destination using random exploration.

## Running Experiments from a Config

For more complex workflows you can specify a sequence of training and
transfer tasks in a YAML file and execute them via `config_runner.py`.
Pass the file path using `--config`:

```bash
python config_runner.py --config example_experiment.yaml
```

`example_experiment.yaml` provides a sample configuration showing how to train a
model and then transfer it to multiple games.


## Customizing World Modeling Prompts

The directory `world_modeling_prompts/` contains the default text prompts used when
initializing or revising the learned world model. To override these prompts for a
specific game, create a subdirectory named after the game inside
`world_modeling_prompts/` and place `initialize_world_model.txt` and
`revise_world_model.txt` files there.

During `TheoryCoderAgent.run()` the agent loads prompts from
`world_modeling_prompts/<game_name>/` if it exists, otherwise it falls back to
the default files at the top level.

Example layout overriding the Sokoban prompts:

```
world_modeling_prompts/
├── initialize_world_model.txt
├── revise_world_model.txt
└── sokoban/
    ├── initialize_world_model.txt
    └── revise_world_model.txt
```

With this directory present the agent will use the Sokoban-specific prompts
whenever the Sokoban environment is run.

## Customizing PDDL Initialization Prompts

The directory `abstraction_prompts/` contains `init_pddl_files.txt` which the
agent uses to synthesize the initial domain and problem files.  Set the
`use_custom_pddl_prompts` option to `true` in your YAML config to allow
game-specific versions named `<game>_init_pddl_files.txt` to override the
default prompt.

This repository ships customized prompts for BabyAI and Sokoban:

```
abstraction_prompts/babyai_init_pddl_files.txt
abstraction_prompts/sokoban_init_pddl_files.txt
```

The BabyAI prompt also includes the environment's mission, while the Sokoban
prompt states that the mission is to push the boxes into the holes.

## Customizing Predicate Initialization Prompts

The directory `abstraction_prompts/` also contains `init_python_predicates.txt`
which the agent uses to synthesize `predicates.py`.  With
`use_custom_pddl_prompts` enabled, game-specific versions named
`<game>_init_python_predicates.txt` override the default prompt.

This repository provides a specialized prompt for the carrying mechanic in the
BabyAI environment:

```
abstraction_prompts/babyai_init_python_predicates.txt
```

## Configuring Fast-Downward

The planner requires the [Fast-Downward](https://www.fast-downward.org/) script.
Create a `downward_config.yaml` file in the repository root to specify its
location:

```yaml
fast_downward_path: /absolute/path/to/fast-downward.py
```

`TheoryCoderAgent` automatically reads this file (or the `FAST_DOWNWARD_PATH`
environment variable) when invoking the planner. If neither is provided the
agent expects `fast-downward.py` to be available on your `PATH`.

## Centralized Domain & Predicates

By default the agent merges each game's generated `domain.pddl` and
`predicates.py` into a shared copy under the experiment directory.  Set the
`centralize_files` option to `false` (or pass `centralize_files=False` when
constructing `TheoryCoderAgent`) to keep per-game files independent and skip
updates to the shared versions.

The predicate file merge normally keeps existing function definitions. Set the
`replace_predicates` option to `true` to overwrite any predicate functions with
the same name when new code is generated.


At each prompt enter an action (e.g. `up`, `down`, `left`, `right`). The script displays the predicted next state from the world model, the actual state from the environment, and whether the predicate is satisfied.
