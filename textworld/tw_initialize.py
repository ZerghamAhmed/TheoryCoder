import os
import csv
import textworld.gym
import time
from groq import Groq
import re

# Initialize the Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# List of models and environments
models = ["llama3-8b-8192"]

environments = [
    {"id": "env_easy", "difficulty": 1, "game_path": "tw_games/easy.z8"},
]

# Configure EnvInfos with desired parameters
request_infos = textworld.EnvInfos(
    description=True,
    inventory=True,
    entities=True,
    admissible_commands=True,
    facts=True,
    objective=True,
    verbs=True,
    win_facts=True,
    fail_facts=True
)

# CSV file to store results
csv_file = "experiment_results_symb.csv"

# Check if the CSV file exists and prepare it accordingly
file_exists = os.path.exists(csv_file)

with open(csv_file, mode="a", newline="") as file:  # Open file in append mode
    writer = csv.writer(file)
    if not file_exists:
        # Write header row if file does not exist
        writer.writerow(["Model", "Environment", "Difficulty", "Run", "Moves", "Score"])

def extract_code_from_response(response):
    # Use a regular expression to extract the Python code within ```Python ``` tags
    code_match = re.search(r'```Python(.*?)```', response, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    else:
        return None

def overwrite_world_model(new_code):
    # Path to the current worldmodel.py file
    world_model_path = "worldmodel.py"
    
    # Write the new code to the worldmodel.py file
    with open(world_model_path, 'w') as file:
        file.write(new_code)

# Run experiments
for model in models:
    print(f' --- Experiments with Model: {model} ---')
    for env in environments:
        print(f' --- Environment: {env} ---')
        # Register the game and create the environment
        env_id = textworld.gym.register_game(env["game_path"], request_infos, max_episode_steps=20)
        textworld_env = textworld.gym.make(env_id)

        for run in range(1, 2):  # Run 1 trial per model per environment
            print(f'---- Trial: {run} ----')
            obs, infos = textworld_env.reset()
            textworld_env.render()

            score, moves, done = 0, 0, False
            messages = []  # Store interaction history for the model

            while not done:
                # Prepare the prompt for the LLM by including entities and admissible commands
                entities_list = ', '.join(infos["entities"])
                goal_natural_language = infos["objective"]
                # goal_fluent = infos["win_facts"]
                # goal_fluent = "\n".join(map(str, infos["win_facts"][0]))
                # goal_fluent = "\n".join(map(str, [goal for condition in infos["win_facts"] for goal in condition]))

                goal_fluent = [
                    str(fact).replace("Proposition", "").replace("Variable", "").replace("(", "").replace(")", "").replace("'", "").replace(",", ":")
                    for condition in infos["win_facts"]
                    for fact in condition
                ]


                commands_list = '\n  '.join(infos["admissible_commands"])
                inventory_list = infos['inventory']

                facts_list = "\n".join(map(str, infos["facts"]))

                prompt_content = (
                    f"Build a python function that serves as a transition model for this environment. "
                    f"Here is some information for your goal, state, etc. in this environment. "
                    f"You will need to use the symbolic version in your transition function.\n\n"
                    f"Goal (in natural language): {goal_natural_language}\n\n"
                    f"Goal (symbolic): {goal_fluent}\n\n"
                    f"State (in natural language): {obs}\n\n"
                    f"State (symbolic):\n{facts_list}\n\n"
                    f"Entities in the room: {entities_list}\n\n"
                    f"Inventory: {inventory_list}\n\n"
                    f"Possible Actions:\n  {commands_list}\n\n"
                    f"Please write the Python code for the transition model below:\n\n"
                    f"RESPONSE FORMAT:\n"
                    f"```Python\n"
                    f"\n"
                    f"def transition_model(state, action):\n"
                    f"\n"
                    f"\n\tReturn State\n"
                    f"```"
                )

                print("PROMPT BELOW:")
                print("\n")
                print(prompt_content)

                breakpoint()

                # Use the model to generate the Python world model
                if moves == 0:
                    messages = [{"role": "user", "content": prompt_content}]
                else:
                    messages.append({"role": "user", "content": prompt_content})

                # Check message length and manage context size
                if len(messages) > 20:
                    # Keep only the most recent 20 messages
                    messages = messages[-20:]

                # Call the language model to generate the transition model code
                chat_completion = client.chat.completions.create(
                    messages=messages,
                    model=model
                )
                response = chat_completion.choices[0].message.content.strip()

                print(response)

                # Extract the Python code from the response
                world_model_code = extract_code_from_response(response)

                if world_model_code:
                    # Save the world model to a Python file
                    overwrite_world_model(world_model_code)
                    print("World model saved to worldmodel.py.")
                else:
                    print("No valid Python code found in the response.")

                # Break the loop after saving the world model
                breakpoint()

            textworld_env.close()

print(f"Experiment results saved to {csv_file}.")
