initialize_world_model_prompt = \
"""You are an AI agent that must come up with a transition model of the game you are playing. 

A BFS low-level planner that will use your synthesized transition model to find the low-level actions that will allow you to win levels of the game.

You are also given state transition after executing random actions that will help as well.
Note that if there is no change returned after doing that action, it means that moving was prevented somehow such as by an obstacle. 

DESCRIPTION OF DOMAIN:

In this domain, you control the avatar and need to reach the goal.

If you touch a trap you will die.


CURRENT STATE:

{current_state}

ACTION SPACE:

{actions_set}

Replay Buffer (last {num_random_actions} transitions):

{errors_from_world_model}

UTILS:

{utils}


RESPONSE FORMAT:

- Make sure you use .get() to access the dictionary to avoid key errors!
For example:
avatar_pos = new_state.get('avatar') to get avatar pos 
cake_pos = new_state.get('cake') to get cake pos


```python

# make sure to include these import statements
from utils import directions

def transition_model(state, action):


	Return State

```
"""

revise_world_model_prompt = \
""" You are an AI agent that must come up with a model of the game you are playing. This model you are making of the game
will be a python program that captures the logic and mechanics of the game. You have begun this world model, but it did not capture everything. 
Below is your current world model, the action space, and the state transition that your transition model handled wrong.
The state transition (inital state, action, next state) will be followed by a section detailing your prediction errors.
If the prediction errors is blank it means your world model correctly modeled that transition.

In order to craft the world model and get this state transition you explored your environment with an EXPLORATION PLAN.
The state transitions belonging to an EXPLORATION PLAN will be written below it.
Note this exploration is a high level plan and the transitions related to it
are carrying out this high level plan by 
executing actions in the ACTION SPACE {actions_set}

Pay close attention to what is involved and modify your transition model to be able to handle this.

DESCRIPTION OF DOMAIN:

In this domain, you control the avatar and need to reach the goal.

If you touch a trap you will die.


NOTES:

Feel free to also explain your thinking outside of the markup tags, but know that I will only use the code inside the markup tags. 

The exploration plans are set up to help guide you to your overall goal. 

Make sure you use .get() to access the dictionary to avoid key errors!

For example:
avatar_pos = new_state.get('avatar') to get avatar pos 
cake_pos = new_state.get('cake') to get cake pos

ACTION SPACE:

{actions_set}

CURRENT WORLD MODEL:

{world_model_str}


ERRORS FROM WORLD MODEL:

{errors_from_world_model}

UTILS:

{utils}


RESPONSE FORMAT (make sure to include your code in markup tags):

- Make sure that you return the correct state for example if you made a deepcopy of the state and modify the deep copy then return the new_state
- If you modify the state directly then return the state instead of new_state

```Python

# make sure to include these import statements
from copy import deepcopy
from utils import directions

def transition_model(state, action):


        Return State

```
"""

debug_model_prompt = """You are an AI agent that must come up with a model of the game you are playing. This model you are making of the game
will be a python program that captures the logic and mechanics of the game. There has been an execution error in your world model.

Please fix your world model code so that this execution error is fixed. You are given the action space, state format, world model as context.

Try to make your world model as general as possible and account for possible cases that may arise in the future!
Also DO NOT make changes to "won" in the state dictionary since that will happen outside of the world model.


ACTION SPACE:

{actions_set}


UTILS:

{utils}

CURRENT WORLD MODEL:

{world_model_str}

DEBUG:

state = {state}
model(state, {action})

ERROR:

{error}

RESPONSE FORMAT (make sure to include your code in markup tags):

```Python

# make sure to include these import statements
from copy import deepcopy
from utils import directions

def transition_model(state, action):


	Return State

```

"""

prune_exploration_prompt = """You are an AI agent that must come up with a model of the game you are playing. This model you are making of the game
will be a python program that captures the logic and mechanics of the game. There has been an execution error in your world model.

You need to carry out an exploratory goal that will help you understand what your model is missing.

You are given the following suggestion for exploratory plans. Which one of this is the most likely one that you should carry out?

Think about the current state of the game and the current world model you have. Also include an explanation.

Notes:

Think about what types of interactions are missing in your world model. For example, try colliding into different objects
or try pushing other objects into others. Think deeply about which of these interactions you have not seen before.

SUGGESTED EXPLORATORY PLANS: 

{suggested_exploratory_plans}

CURRENT STATE:

{current_state}

CURRENT WORLD MODEL:

{world_model_str}

RESPONSE FORMAT (make sure to include your code in Python markup tags):

```Python

# just an example DO NOT ouput this
[move_to agent_1 place_2]

```

Explanation: Example explanation of why you chose this plan.

"""

execute_exploratory_plan_prompt = """You are an AI agent that must come up with a model of the game you are playing. This model you are making of the game
will be a python program that captures the logic and mechanics of the game. There has been an execution error in your world model.

You need to carry out an exploratory goal that will help you understand what your model is missing.

You are given the following suggestion for exploratory plans. However, these suggestions are high-level 
and you cannot execute them in the game. The actual executable action set for the game is 
["left", "right", "forward", "pickup", "drop", "toggle"] 

Please give the actions that will allow you to execute each of them.

Think about the current state of the game and the current world model you have. Also include an explanation.

Notes:

Please also write an explanation for why your low-level action plan satisfies the high-level suggested exloratory plan.
Please also relate it to how it can uncover the errors in the current incorrect world model.

SUGGESTED EXPLORATORY PLANS: 

{suggested_exploratory_plans}

CURRENT STATE:

{current_state}

CURRENT WORLD MODEL (NOT CORRECT):

{world_model_str}

RESPONSE FORMAT (make sure to include your code in Python markup tags):

```Python

# exploratory plan 1: open brown_door 
["forward", "forward" "toggle"]

# exploratory plan 2: drop white_key 
["left", "left" "drop"]

# exploratory plan 2: drop white_key 
["left", "left" "drop"]

# exploratory plan 3: pickup white_key
["forward", "right", "pickup"]

```

Explanation: Example explanation of why you chose this low-level action plan for each exploratory plan.
Explanation for relating to world model correction: Explain how it can uncover the errors in world model.

"""
