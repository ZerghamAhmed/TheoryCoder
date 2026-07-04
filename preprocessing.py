import re
import os
import inspect
import importlib.util


class PredicateExecutionError(Exception):
    """Exception raised when a predicate function fails during evaluation."""

    def __init__(self, predicate_name, args, original_exception):
        super().__init__(f"{predicate_name} failed: {original_exception}")
        self.predicate_name = predicate_name
        self.args = args
        self.original_exception = original_exception

# Path to the predicates file can be configured via environment variable
PREDICATES_FILE_PATH = os.getenv("TC_PREDICATES_FILE", "predicates.py")


def extract_predicates_section(domain_lines):
    predicates = {}
    inside_predicates = False
    for line in domain_lines:
        line = line.strip()
        if line.startswith("(:predicates"):
            inside_predicates = True
            continue
        if inside_predicates:
            if line.startswith(")"):
                break
            match = re.match(r"\(([\w-]+)\s+(.*?)\)", line)
            if match:
                name = match.group(1)
                args = re.findall(r"\?[\w\-]+", match.group(2))
                predicates[name] = len(args)
    return predicates








def parse_logical_expression(expression):
    tokens = re.findall(r'\(|\)|[\w-]+|not', expression)
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






def _tokenize_pddl(text):
    """Yield PDDL tokens. Skips ; line comments. Treats ( and ) as separate tokens.

    Atoms are runs of non-whitespace, non-paren, non-semicolon characters
    (so ?vars, :keywords, identifiers with - or _ all parse as one atom).
    """
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
        elif c == ';':
            while i < n and text[i] != '\n':
                i += 1
        elif c in '()':
            yield c
            i += 1
        else:
            j = i
            while j < n and not text[j].isspace() and text[j] not in '();':
                j += 1
            yield text[i:j]
            i = j


def _parse_sexpr(tokens, pos):
    """Parse one s-expression starting at tokens[pos]. Returns (node, next_pos).

    Atoms become strings; lists become nested Python lists.
    """
    if pos >= len(tokens):
        raise ValueError("PDDL parse error: unexpected end of input")
    tok = tokens[pos]
    if tok == '(':
        result = []
        pos += 1
        while pos < len(tokens) and tokens[pos] != ')':
            sub, pos = _parse_sexpr(tokens, pos)
            result.append(sub)
        if pos >= len(tokens):
            raise ValueError("PDDL parse error: unbalanced parens")
        return result, pos + 1  # consume ')'
    if tok == ')':
        raise ValueError("PDDL parse error: unexpected ')'")
    return tok, pos + 1


def _strip_qmark(node):
    """Strip leading '?' from atoms in a parsed s-expression tree.

    Old line-based parser used re.findall(r'\\(|\\)|[\\w-]+|not', ...) which
    silently dropped '?' from all atoms in preconditions/effects (because '?'
    isn't in \\w). Downstream code (auto_generate_predicate_arg_mapping, etc.)
    relies on that contract — args have no '?' prefix; parameters keep it.
    """
    if isinstance(node, list):
        return [_strip_qmark(c) for c in node]
    if isinstance(node, str) and node.startswith('?'):
        return node[1:]
    return node


def parse_domain_file(domain_file):
    """Robust PDDL domain parser.

    Returns the same shape as the original line-based parser:
        {action_name: {'parameters': [vars], 'preconditions': sexpr, 'effects': sexpr}}

    Handles multi-line preconditions/effects, ; comments, arbitrary nesting,
    and balanced parens — anything the old line-by-line scanner couldn't.
    """
    with open(domain_file, 'r') as file:
        text = file.read()

    tokens = list(_tokenize_pddl(text))
    if not tokens:
        return {}

    try:
        top, _ = _parse_sexpr(tokens, 0)
    except ValueError as e:
        print(f"[parse_domain_file] ERROR: {e}")
        return {}

    domain_data = {}
    if not isinstance(top, list):
        return domain_data

    # top is (define (domain NAME) (:requirements ...) (:types ...) (:predicates ...) (:action ...) ...)
    for node in top:
        if not (isinstance(node, list) and len(node) >= 2 and node[0] == ':action'):
            continue
        action_name = node[1]
        params, preconds, effects = [], [], []
        i = 2
        while i < len(node) - 1:
            key, val = node[i], node[i + 1]
            if key == ':parameters':
                if isinstance(val, list):
                    params = [t for t in val if isinstance(t, str) and t.startswith('?')]
            elif key == ':precondition':
                preconds = _strip_qmark(val)
            elif key == ':effect':
                effects = _strip_qmark(val)
            i += 2
        domain_data[action_name] = {
            'parameters': params,
            'preconditions': preconds,
            'effects': effects,
        }

    return domain_data


def auto_generate_predicate_arg_mapping(domain_file_path):
    if not os.path.exists(domain_file_path):
        print(f"Domain file '{domain_file_path}' not found. Predicate mappings will be empty.")
        return {}
    with open(domain_file_path, 'r') as f:
        lines = f.readlines()


    # Step 1: Parse predicates section to get arities
    arity_map = extract_predicates_section(lines)
    mapping = {k: [] for k in arity_map}


    # Step 2: Process actions
    domain_data = parse_domain_file(domain_file_path)
    for action in domain_data.values():
        param_list = action["parameters"]
        for section in ["preconditions", "effects"]:
            section_data = action[section]

            # Handle case where section_data is a single predicate like ['open', 'd']
            # vs a compound expression like ['and', ['carrying', 'a', 'k']]
            # or a list of predicates
            if isinstance(section_data, list) and len(section_data) > 0:
                first_elem = section_data[0]
                # If first element is a string that's not 'and'/'or'/'not',
                # this is a single predicate - wrap it in a list
                if isinstance(first_elem, str) and first_elem not in ['and', 'or', 'not']:
                    section_data = [section_data]

            for raw in section_data:
                # Extract actual predicate call
                if isinstance(raw, list) and len(raw) > 0 and raw[0] == "not":
                    if isinstance(raw[1], list):
                        pred_name = raw[1][0]
                        args = raw[1][1:]
                    else:
                        continue
                elif isinstance(raw, list) and len(raw) > 0:
                    pred_name = raw[0]
                    args = raw[1:]
                else:
                    continue


                # Only match arity-matching predicates
                if pred_name in arity_map and len(args) == arity_map[pred_name]:


                    # breakpoint()
                    candidate = ["?" + arg for arg in args if f"?{arg}" in param_list]


                    if candidate not in mapping[pred_name]:
                        mapping[pred_name].append(candidate)
   
    return mapping






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
    if not os.path.exists(predicates_file_path):
        print(f"Predicates file '{predicates_file_path}' not found.")
        predicate_functions = {}
        return

    module_name = "predicates"
    spec = importlib.util.spec_from_file_location(module_name, predicates_file_path)
    predicates_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(predicates_module)


    # Extract all functions from the module
    predicate_functions = {
        name: func
        for name, func in inspect.getmembers(predicates_module, inspect.isfunction)
    }


    # print(f"Loaded predicates: {list(predicate_functions.keys())}")




# Global dictionary to store predicate functions.  Call `initialize_predicates`
# explicitly after configuring `TC_PREDICATES_FILE`.

# Use environment variable to set initial domain file if provided
current_domain_file = os.getenv("TC_DOMAIN_FILE")

# Start with an empty mapping until a domain file is explicitly loaded
predicate_arg_mapping = {}

if current_domain_file:
    predicate_arg_mapping = auto_generate_predicate_arg_mapping(current_domain_file)


def set_domain_file(domain_file_path: str) -> None:
    """Set the current domain file and regenerate predicate mappings."""
    global current_domain_file, predicate_arg_mapping
    current_domain_file = domain_file_path
    predicate_arg_mapping = auto_generate_predicate_arg_mapping(current_domain_file)
    print(f"[DEBUG] set_domain_file called with: {domain_file_path}")
    print(f"[DEBUG] predicate_arg_mapping now: {predicate_arg_mapping}")


# breakpoint()


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


















def operator_extractor(domain_file, subplan):
    print(f"[DEBUG] operator_extractor called with domain_file = {domain_file!r}, subplan = {subplan!r}")


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
    # Reload predicates using the latest path from the environment, falling
    # back to the previously configured path. This ensures that newly
    # generated predicate files are loaded correctly.
    latest_pred_file = os.environ.get("TC_PREDICATES_FILE", PREDICATES_FILE_PATH)
    initialize_predicates(latest_pred_file)
    # Use the globally configured domain file
    global predicate_arg_mapping




   
    # if operators['operator'] == 'put_next_to':
    #     breakpoint()


    for predicate in predicates:
        # print(f"Evaluating predicate: {predicate}")


        # if predicate == "blocking":
        #     breakpoint()


        # if predicate == "put_next_to":
        #     breakpoint()


        is_negated = predicate.startswith("not ")
        predicate_name = predicate[4:] if is_negated else predicate

        # Convert hyphens to underscores for Python function compatibility
        # PDDL allows hyphens but Python function names cannot have hyphens
        predicate_name_py = predicate_name.replace("-", "_")

        if predicate_name_py not in predicate_functions:
            raise ValueError(f"Unknown predicate: {predicate_name} (tried: {predicate_name_py})")


        # Get all possible parameter sets for the predicate (use original name for mapping)
        possible_param_sets = predicate_arg_mapping.get(predicate_name, [])
        args = None


        for param_set in possible_param_sets:
            try:
                # Attempt to resolve all parameters in the set
                args = [grounding[param] for param in param_set]
                break  # Stop once we successfully resolve a parameter set
            except KeyError:
                continue  # Try the next parameter set


        # Fallback: if no mapping found, try using all action parameters in order
        # This handles cases where predicates weren't captured in the domain file mapping
        if args is None and possible_param_sets == []:
            param_list = list(grounding.values())
            # Try to call the predicate with all available parameters
            # (truncated to reasonable arity, typically 1-4 parameters)
            for arity in range(1, min(len(param_list) + 1, 5)):
                try:
                    test_args = param_list[:arity]
                    # Test if predicate can be called with this arity
                    result_test = predicate_functions[predicate_name_py](state, *test_args)
                    args = test_args
                    break
                except TypeError:
                    # Wrong number of arguments, try next arity
                    continue
                except Exception:
                    # Other errors, try next arity
                    continue


        if args is None:
            # breakpoint()
            print(f"[DEBUG] predicate_arg_mapping = {predicate_arg_mapping}")
            print(f"[DEBUG] possible_param_sets for '{predicate_name}' = {possible_param_sets}")
            raise KeyError(f"No matching parameter set for {predicate_name}. Grounding: {grounding}. Available predicates: {list(predicate_functions.keys())}")
        # breakpoint()
        # Call the predicate with state and resolved arguments
        try:
            result = predicate_functions[predicate_name_py](state, *args)
        except Exception as e:
            raise PredicateExecutionError(predicate_name_py, args, e) from e
       
        # print(*args)
        if is_negated:
            result = not result


        # if operators["operator"] == "put_next_to":
        #     breakpoint()


        results.append(result)


    # if operators["operator"] == "put_next_to":
    #     breakpoint()


    # if operators["operator"] == "move_to":
    #     breakpoint()




    # print("Predicate evaluations:", results)
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
                param_mappings = predicate_arg_mapping.get(predicate_name, [])
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



