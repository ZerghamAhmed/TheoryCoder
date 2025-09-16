import re
from predicates import *
import inspect
import predicates
import importlib.util


def initialize_predicates(predicates_file_path):
    """
    Load predicates from a given predicates.py file and store them globally.


    Args:
        predicates_file_path (str): Path to the predicates.py file.


    Returns:
        None
    """
    global predicate_functions


    # Load the predicates module dynamically
    module_name = "predicates"
    spec = importlib.util.spec_from_file_location(module_name, predicates_file_path)
    predicates_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(predicates_module)


    # Extract all functions from the module
    predicate_functions = {
        name: func
        for name, func in inspect.getmembers(predicates_module, inspect.isfunction)
    }


    print(f"Loaded predicates: {list(predicate_functions.keys())}")




# Global dictionary to store predicate functions
initialize_predicates("predicates.py")


precondition_param_mapping = {
    "controllable": [["?obj"]],
    "overlapping": [["?obj", "?to"]],
    "carrying": [["?item"], ["?obj"], ["?key"]],  # Multiple alternatives
    "inventory_full": [[]],  # No parameters needed beyond `state`
    "locked": [["?door"]],
    "open_door": [["?door"]],
    "unlocks": [["?key", "?door"]],
    "clear": [["?door"]],
    "next_to": [["?item", "?adjacent"], ["?door", "?obj"], ["?obj", "?to"]],
    "blocking": [["?obj", "?door"]],
    "agent_moved_away": [["?door"]],
    "stationary": [["?adjacent"]],
    "not_near_door": [["?obj"]],
    "found": [["?key"]],
    "unfound": [["?box"]],  # Explicit mapping for negated found
    "went_through": [["?agent", "?door"]],
    "safe_to_collect": [["?diamond"]],
    "collected": [["?diamond"]],
    "diamond_bag_full": [[]],  # No parameters needed beyond `state`
    "no_more_boxes": [["?boxes"]],  # No parameters needed beyond `state`
    "unstored_box": [["?box"]],  # No parameters needed beyond `state`
    "boxes_stuck": [[]],  # No parameters needed beyond `state`





}




# explicit_type_mapping = {
#     "overlapping": {
#         "?obj": ["red_agent"],
#         "?to": ["red_ball", "blue_ball", "green_ball", "yellow_ball"]
#     },
#     "open_door": {
#         "?door": ["closed_red_door", "open_purple_door", "locked_yellow_door"]
#     },
#     "unlock": {
#         "?door": ["locked_red_door", "locked_yellow_door"],
#         "?key": ["red_key", "yellow_key"]
#     },
#     "pickup": {
#         "?item": ["red_ball", "blue_box", "green_key"]
#     },
#     # Add other predicates as needed
# }


def domain_specific_type_system_mapping_BABYAI(state):
    """
    Automatically generate the explicit type mapping based on the domain-specific rules.


    Args:
        state (dict): The current game state.


    Returns:
        dict: The explicit type mapping for all predicates.
    """
    explicit_type_mapping = {}


    # 1. "controllable"
    explicit_type_mapping["controllable"] = {
        "?obj": ["red_agent"]
    }


    # 2. "overlapping"
    explicit_type_mapping["overlapping"] = {
        "?obj": ["red_agent"],
        "?to": [entity for entity in state.keys() if entity != "grey_wall"]
    }


    # 3. "open_door"
    explicit_type_mapping["open_door"] = {
        "?door": [entity for entity in state.keys() if entity.endswith("_door")]
    }


    # 4. "unlocks"
    explicit_type_mapping["unlocks"] = {
        "?door": [entity for entity in state.keys() if entity.startswith("locked_") and entity.endswith("_door")],
        "?key": [entity for entity in state.keys() if entity.endswith("_key")]
    }


    # 5. "pickup"
    explicit_type_mapping["pickup"] = {
        "?item": [entity for entity in state.keys() if entity.endswith(("_ball", "_box", "_key"))]
    }


    # 6. "carrying"
    explicit_type_mapping["carrying"] = {
        "?item": [entity for entity in state.keys() if entity.endswith(("_ball", "_box", "_key"))],
        "?obj": [entity for entity in state.keys() if entity.endswith(("_ball", "_box", "_key"))],
        "?key": [entity for entity in state.keys() if entity.endswith("_key")]
    }


    # 7. "locked"
    explicit_type_mapping["locked"] = {
        "?door": [entity for entity in state.keys() if entity.endswith("_door")]
    }


    # 8. "clear"
    explicit_type_mapping["clear"] = {
        "?door": [entity for entity in state.keys() if entity.endswith("_door")]
    }


    # 9. "next_to"
    explicit_type_mapping["next_to"] = {
        "?item": [entity for entity in state.keys() if entity != "grey_wall"],
        "?adjacent": [entity for entity in state.keys() if entity != "grey_wall"]
    }


    # 10. "blocking"
    explicit_type_mapping["blocking"] = {
        "?obj": [entity for entity in state.keys() if entity.endswith(("_box", "_key", "_ball"))],
        "?door": [entity for entity in state.keys() if entity.endswith("_door")]
    }


    # 11. "agent_moved_away"
    explicit_type_mapping["agent_moved_away"] = {
        "?door": [entity for entity in state.keys() if entity.endswith("_door")]
    }


    # 12. "not_near_door"
    explicit_type_mapping["not_near_door"] = {
        "?obj": [entity for entity in state.keys() if entity.endswith(("_key", "_box", "_ball"))]
    }


    # 13. "clear"
    explicit_type_mapping["clear"] = {
        "?door": [entity for entity in state.keys() if entity.endswith("_door")]
    }


    # 14. "found"
    explicit_type_mapping["found"] = {
        "?key": [entity for entity in state.keys() if entity.endswith("_key")]
    }


    # 13. "unfound"
    explicit_type_mapping["unfound"] = {
        "?box": [entity for entity in state.keys() if entity.endswith("_box")]
    }


    # breakpoint()


    return explicit_type_mapping








def load_predicate_functions(module):
    """
    Load all functions from the given module dynamically.


    Args:
        module: The module from which to extract functions.


    Returns:
        dict: A dictionary mapping function names to their implementations.
    """
    return {
        name: func
        for name, func in inspect.getmembers(module, inspect.isfunction)
    }


def preprocess_subplan(subplan):
    """
    Processes a given subplan string based on the specified action type.
    Depends on Domain file and level of abstraction of plans.


    The function supports two actions:
    1. 'form_rule': Removes the numeric suffix from each parameter.
    2. 'move_to': Extracts the base object name and converts the numeric suffix to a zero-based index.


    Parameters:
    subplan (str): A string representing the action and its parameters. The format is
                   "action param1 param2 ...", where each parameter may have a numeric
                   suffix separated by an underscore.


    Returns:
    list: A list of processed parameters.
          - For 'form_rule', it returns a list of strings with numeric suffixes removed.
            Example:
                Input: 'form_rule flag_word is_word win_word'
                Output: ['flag_word', 'is_word', 'win_word']
          - For 'move_to', it returns a list alternating between the base object names
            and their corresponding zero-based indices.
            Example:
                Input: 'move_to baba_obj flag_obj'
                Output: ['baba_obj', 'flag_obj']
    """


    action, *params = subplan.split()
    # formatted_params = []






    # if action in ['form_rule', 'break_rule']:
    #     formatted_params = [param for param in params]


    # if action in ['move_to', 'push_to']:
    #     formatted_params = [param for param in params]


    return [param for param in params]


def parse_domain_file(domain_file):
    with open(domain_file, 'r') as file:
        lines = file.readlines()
   
    domain_data = {}
    current_action = None
    for line in lines:
        line = line.strip()
       
        if line.startswith("(:action"):
            current_action = line.split()[1]
            domain_data[current_action] = {"parameters": [], "preconditions": [], "effects": []}
       
        elif current_action and line.startswith(":parameters"):
            parameters = re.findall(r'\?\w+', line)
            domain_data[current_action]["parameters"] = parameters
       
        elif current_action and line.startswith(":precondition"):
            precondition_str = line.replace(":precondition", "").strip()
            preconditions = parse_logical_expression(precondition_str)
            domain_data[current_action]["preconditions"] = preconditions
       
        elif current_action and line.startswith(":effect"):
            effect_str = line.replace(":effect", "").strip()
            effects = parse_logical_expression(effect_str)
            domain_data[current_action]["effects"] = effects
       
        elif current_action and line == ")":
            current_action = None
   
    return domain_data


def parse_logical_expression(expression):
    tokens = re.findall(r'\(|\)|\w+|not', expression)
    stack = []
    current = []


    for token in tokens:
        if token == '(':
            stack.append(current)
            current = []
        elif token == ')':
            if stack:
                parent = stack.pop()
                parent.append(current)
                current = parent
        else:
            current.append(token)
   
    return current[0] if len(current) == 1 else current  # Flatten top-level nesting














def operator_extractor(domain_file, subplan):
    domain_data = parse_domain_file(domain_file)
    operator = subplan.split()[0]
   
    if operator in domain_data:
        parameters = domain_data[operator]["parameters"]
        preconditions = extract_predicates(domain_data[operator]["preconditions"])
        effects = extract_predicates(domain_data[operator]["effects"])
        # breakpoint()
        # depends on abstraction level of domain file used
        formatted_args = preprocess_subplan(subplan)


        # if operator == 'unblock':
        #     breakpoint()


        # if operator == 'move_to':
        #     breakpoint()
       
        return {"operator": operator, "parameters": parameters, "preconditions": preconditions, "effects": effects, "grounding_Python": formatted_args}
    else:
        raise ValueError(f"Operator {operator} not found in domain file.")
   
def extract_predicates(conditions):
    predicates = []


    if isinstance(conditions, list):
        if conditions[0] == 'not':  # Handle negation
            predicates.append(f"not {conditions[1][0]}")  # Add 'not' with predicate name
        elif conditions[0] in ['and', 'or']:  # Handle logical operators
            for sub_condition in conditions[1:]:
                predicates.extend(extract_predicates(sub_condition))  # Recurse
        else:  # Direct predicate name
            predicates.append(conditions[0])
   
    return predicates






def checker(state, predicates, operators):
    results = []
    grounding = {param: value for param, value in zip(operators["parameters"], operators["grounding_Python"])}
   
    # if operators['operator'] == 'put_next_to':
    #     breakpoint()




    for predicate in predicates:
        print(f"Evaluating predicate: {predicate}")


        # if predicate == "blocking":
        #     breakpoint()


        # if predicate == "put_next_to":
        #     breakpoint()


        is_negated = predicate.startswith("not ")
        predicate_name = predicate[4:] if is_negated else predicate


        if predicate_name not in predicate_functions:
            raise ValueError(f"Unknown predicate: {predicate_name}")


        # Get all possible parameter sets for the predicate
        possible_param_sets = precondition_param_mapping.get(predicate_name, [])
        args = None


        for param_set in possible_param_sets:
            try:
                # Attempt to resolve all parameters in the set
                args = [grounding[param] for param in param_set]
                break  # Stop once we successfully resolve a parameter set
            except KeyError:
                continue  # Try the next parameter set


        if args is None:
            # breakpoint()
            raise KeyError(f"No matching parameter set for {predicate_name}. Grounding: {grounding}")


        # Call the predicate with state and resolved arguments
        result = predicate_functions[predicate_name](state, *args)
        print(*args)
        if is_negated:
            result = not result


        # if operators["operator"] == "put_next_to":
        #     breakpoint()


        results.append(result)


    # if operators["operator"] == "put_next_to":
    #     breakpoint()


    # if operators["operator"] == "move_to":
    #     breakpoint()




    print("Predicate evaluations:", results)
    return all(results)






def is_and_expression(subplan):
    """Check if the subplan is an AND expression."""
    return subplan.startswith('AND(')


def evaluate_and_expression(domain_file, expression, state):
    """
    Evaluates a string of the form AND(subplan1, subplan2) and returns the AND result.
   
    Args:
        domain_file (str): Path to the PDDL domain file.
        expression (str): The AND expression containing two subplans.
        state (dict): The current game state.


    Returns:
        bool: True if both subplans are satisfied, False otherwise.
    """
    # Modify the regex to extract subplans without quotes
    match = re.match(r'AND\((.+?),\s*(.+?)\)', expression)
    if not match:
        raise ValueError("Expression is not in the correct AND format.")


    # Extract the two subplans
    subplan_1 = match.group(1).strip()
    subplan_2 = match.group(2).strip()


    # Extract operator and preconditions for the first subplan
    operator_1 = operator_extractor(domain_file, subplan_1)
    preconditions_1 = operator_1['preconditions']
    result_1 = checker(state, preconditions_1, operator_1)


    # Extract operator and preconditions for the second subplan
    operator_2 = operator_extractor(domain_file, subplan_2)
    preconditions_2 = operator_2['preconditions']
    result_2 = checker(state, preconditions_2, operator_2)


    # breakpoint()


    # result_1 = not result


    # Return the AND of both results
    return not result_1 and not result_2




# def enumerate_groundings(domain_file, state):
#     """
#     Enumerate all possible groundings for operators in a domain given a state.


#     Args:
#         domain_file (str): Path to the PDDL domain file.
#         state (dict): Current state dictionary.


#     Returns:
#         dict: A dictionary where keys are operators and values are lists of possible groundings.
#     """
#     domain_data = parse_domain_file(domain_file)
#     groundings = {}


#     # Match entities in state to PDDL types
#     type_mapping = {key: [] for key in ["object", "door"]}


#     for entity in state.keys():
#         if entity.endswith("_door"):
#             type_mapping["door"].append(entity)
#         elif entity not in {"red_agent", "agent_direction", "agent_carrying"}:  # Exclude agent-specific keys
#             type_mapping["object"].append(entity)


#     # Enumerate groundings for each operator
#     for operator, data in domain_data.items():
#         param_types = data["parameters"]


#         # Deduce types from the domain file
#         param_types_cleaned = []
#         for param in param_types:
#             match = re.match(r"\?\w+ - (\w+)", param)
#             param_type = match.group(1) if match else "object"
#             param_types_cleaned.append(param_type)


#         # Generate possible groundings using Cartesian product
#         try:
#             param_entities = [type_mapping[param_type] for param_type in param_types_cleaned]
#             operator_groundings = list(product(*param_entities))
#         except KeyError as e:
#             print(f"Type {e} not found in type mapping for operator {operator}. Defaulting to empty.")
#             operator_groundings = []


#         groundings[operator] = operator_groundings


#     return groundings


from itertools import product


# dictionary method


# def enumerate_possible_subplans(state, domain_file):
#     """
#     Enumerate all possible subplans by generating all combinations of parameters
#     for operators based on the state dictionary.


#     Args:
#         state (dict): The current game state.
#         domain_file (str): The domain file containing operators.


#     Returns:
#         dict: A dictionary where keys are operator names and values are lists of grounded subplans.
#     """
#     # Load domain data
#     domain_data = parse_domain_file(domain_file)


#     # Ignore these keys while generating combinations
#     ignored_keys = {"agent_direction", "agent_carrying"}


#     # Extract valid keys from the state dictionary
#     entities = [
#         key for key in state.keys()
#         if key not in ignored_keys and isinstance(state[key], list) and state[key]  # Ensure the key has positions
#     ]


#     grounded_subplans = {}


#     # Generate grounded subplans for each operator
#     for operator, data in domain_data.items():
#         param_count = len(data["parameters"])  # Number of parameters for the operator


#         # Generate all combinations of entities for the parameters
#         param_combinations = product(entities, repeat=param_count)


#         # Create grounded subplans using string formatting
#         grounded_subplans[operator] = [
#             f"{operator} " + " ".join(params) for params in param_combinations
#         ]


#     return grounded_subplans


def enumerate_possible_subplans(state, domain_file):
    """
    Enumerate all possible subplans for the given state and domain.


    Args:
        state (dict): The current game state.
        domain_file (str): Path to the PDDL domain file.


    Returns:
        list: A flattened list of all grounded subplans in the format:
              ["operator arg1 arg2 ...", ...]
    """
    # Parse the domain file to get operator data
    domain_data = parse_domain_file(domain_file)
   
    # Get all keys from the state (exclude agent-related keys)
    state_keys = [key for key in state.keys() if key not in ['red_agent', 'agent_direction', 'agent_carrying']]


    subplans = []


    # Enumerate grounded subplans for each operator
    for operator, data in domain_data.items():
        param_names = data["parameters"]  # e.g., ['?obj', '?to']


        # Create all combinations of parameters
        grounded_combinations = product(state_keys, repeat=len(param_names))
       
        for grounding in grounded_combinations:
            # Format the subplan as "operator arg1 arg2 ..."
            subplan = f"{operator} " + " ".join(grounding)
            subplans.append(subplan)


    return subplans


def validate_arguments(predicate, args, explicit_type_mapping):
    """
    Validate arguments against the explicit type mapping.


    Args:
        predicate (str): The predicate name (e.g., "overlapping").
        args (dict): A dictionary of argument bindings (e.g., {"?obj": "red_agent", "?to": "red_ball"}).
        explicit_type_mapping (dict): The explicit type mapping dictionary.


    Returns:
        bool: True if the arguments are valid, False otherwise.
    """
    if predicate not in explicit_type_mapping:
        return False  # Unknown predicate


    valid_args = explicit_type_mapping[predicate]
    for arg, value in args.items():
        if arg not in valid_args or value not in valid_args[arg]:
            return False


    return True


def prune_invalid_subplans_TYPE(possible_subplans, type_mapping, domain_file):
    """
    Prune invalid subplans based on the type constraints defined for preconditions.


    Args:
        possible_subplans (list): List of possible subplans (grounded operators).
        type_mapping (dict): Explicit type mapping for each predicate.
        domain_file (str): The domain file to extract operator details.


    Returns:
        list: A list of valid subplans after pruning.
    """
    valid_subplans = []


    for subplan in possible_subplans:
        parts = subplan.split()  # Split into operator and arguments
        operator, *args = parts  # Extract the operator and arguments


        try:
            # Extract operator details from the domain file
            operator_details = operator_extractor(domain_file, subplan)
            preconditions = operator_details["preconditions"]


            # Check if arguments satisfy type constraints for all preconditions
            is_valid = True
            for predicate in preconditions:
                # Extract predicate name and parameters
                is_negated = predicate.startswith("not ")
                predicate_name = predicate[4:] if is_negated else predicate


                # Skip unknown predicates
                if predicate_name not in type_mapping:
                    print(f"Skipping unknown predicate '{predicate_name}' in subplan '{subplan}'.")
                    continue


                # Get type constraints for the predicate
                type_constraints = type_mapping[predicate_name]


                # Resolve arguments for this predicate
                param_mappings = precondition_param_mapping.get(predicate_name, [])
                for param_set in param_mappings:
                    try:
                        resolved_args = {param: args[i] for i, param in enumerate(param_set)}
                        # Validate each argument against its type constraints
                        for param, value in resolved_args.items():
                            if param in type_constraints and value not in type_constraints[param]:
                                print(f"Skipping subplan '{subplan}': Argument '{value}' does not match valid values for '{param}' in predicate '{predicate_name}'.")
                                is_valid = False
                                break
                        if not is_valid:
                            break
                    except IndexError:
                        print(f"Skipping subplan '{subplan}': Argument mismatch in predicate '{predicate_name}'.")
                        is_valid = False
                        break


                if not is_valid:
                    break


            if is_valid:
                valid_subplans.append(subplan)


        except ValueError as e:
            print(f"Error processing subplan '{subplan}': {e}")


    return valid_subplans