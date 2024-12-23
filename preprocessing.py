import re
from predicates import *

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
                Input: 'form_rule flag_word_1 is_word_2 win_word_1'
                Output: ['flag_word', 'is_word', 'win_word']
          - For 'move_to', it returns a list alternating between the base object names 
            and their corresponding zero-based indices.
            Example: 
                Input: 'move_to baba_obj_1 flag_obj_1'
                Output: ['baba_obj', 0, 'flag_obj', 0]
    """

    action, *params = subplan.split()
    formatted_params = []


    if action == 'form_rule':
        formatted_params = [param for param in params]

    if action == 'break_rule':
        formatted_params = [param for param in params]

    if action == 'move_to':
        for param in params:
            # Extract the base name and number using regular expressions
            match = re.match(r'(.+?)_(\d+)$', param)
            if match:
                base_name = match.group(1)  # the object name is match group 1, index is match group 2
                index = int(match.group(2)) - 1  # Convert to zero-based index
                formatted_params.extend([base_name, index])

    # if action == 'push_to':
    #     for param in params:
    #         # Extract the base name and number using regular expressions
    #         match = re.match(r'(.+?)_(\d+)$', param)
    #         if match:
    #             base_name = match.group(1)  # the object name is match group 1, index is match group 2
    #             index = int(match.group(2)) - 1  # Convert to zero-based index
    #             formatted_params.extend([base_name, index])

    if action == 'push_to':
        for param in params:
            match = re.match(r'(.+?)_(\d+)$', param)
            if match:
                base_name = match.group(1)
                index = int(match.group(2)) - 1  # Convert to zero-based index
                formatted_params.extend([base_name, index])


    # elif action in ['move_loc', 'push_to']:
    #     # Extract the coordinates as a list of integers
    #     location = re.search(r'\[(\d+),\s*(\d+)\]', subplan)
    #     if location:
    #         coordinates = [int(location.group(1)), int(location.group(2))]
    #         for param in params:
    #             if '[' in param and ']' in param:
    #                 # Add the coordinates as a list to formatted_params
    #                 formatted_params.append(coordinates)
    #             else:
    #                 # Strip the trailing underscores and digits
    #                 match = re.match(r'(.+?)_(\d+)$', param)
    #                 if match:
    #                     base_name = match.group(1)
    #                     instance_index = int(match.group(2)) - 1  # Convert to zero-based index
    #                     formatted_params.append(base_name)
    #                     formatted_params.append(instance_index)
    #                 else:
    #                     formatted_params.append(re.sub(r'_\d+$', '', param))


    # if action == 'move_loc':
    #     # Extract the coordinates as a list of integers
    #     location = re.search(r'\[(\d+),\s*(\d+)\]', subplan)
    #     if location:
    #         coordinates = [int(location.group(1)), int(location.group(2))]
    #         for param in params:
    #             if '[' in param and ']' in param:
    #                 # Add the coordinates as a list to formatted_params
    #                 formatted_params.append(coordinates)
    #             else:
    #                 # Strip the trailing underscores and digits
    #                 formatted_params.append(re.sub(r'_\d+$', '', param))

    # if action == 'push_to':
    #     # Extract the coordinates as a list of integers
    #     location = re.search(r'\[(\d+),\s*(\d+)\]', subplan)
    #     if location:
    #         coordinates = [int(location.group(1)), int(location.group(2))]
    #         for param in params:
    #             if '[' in param and ']' in param:
    #                 # Add the coordinates as a list to formatted_params
    #                 formatted_params.append(coordinates)
    #             else:
    #                 # Strip the trailing underscores and digits
    #                 formatted_params.append(re.sub(r'_\d+$', '', param))


    return formatted_params


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

        # depends on abstraction level of domain file used
        formatted_args = preprocess_subplan(subplan)

        # if operator == 'push_to':
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
    # if operators['operator'] == 'push_to':
    #     breakpoint()
    for predicate in predicates:

        print("predicate name", predicate)
        
        if predicate == 'control_rule':
            # breakpoint()
            # Extract words from grounding_Python
            word1 = operators["grounding_Python"][0].replace("obj", "word")
            word2 = "is_word"
            word3 = "you_word" 
            # Call rule_formed with these words
            results.append(rule_formed(state, word1, word2, word3))
            # breakpoint()
            if not all(results):
                for entity, coords in state.items():
                    if entity.endswith('_word'):
                        if rule_formed(state, entity, "is_word", "move_word"):
                            print("THERE IS AN AUTOMOVER")
                            results.pop()
                            break
                results.append(True)

        if predicate == 'push_rule':
            # Extract words from grounding_Python
            word1 = operators["grounding_Python"][0].replace("obj", "word")
            word2 = "is_word"
            word3 = "push_word" 
            # Call rule_formed with these words
            results.append(rule_formed(state, word1, word2, word3))

        if predicate == 'rule_formed':
            # Extract words from grounding_Python
            word1, word2, word3 = operators["grounding_Python"]
            print(word1, word2, word3)
            results.append(rule_formed(state, word1, word2, word3))

        if predicate == 'not rule_formed':
            # Extract words from grounding_Python
            word1, word2, word3 = operators["grounding_Python"]
            print(word1, word2, word3)
            results.append(negate(rule_formed(state, word1, word2, word3)))

        if predicate == 'rule_formable':
            # Extract the three words from grounding_Python
            word1, word2, word3 = operators["grounding_Python"]
            results.append(rule_formable(state, word1, word2, word3))

        if predicate == 'rule_breakable':
            # Extract the three words from grounding_Python
            word1, word2, word3 = operators["grounding_Python"]
            # breakpoint()
            results.append(rule_breakable(state, word1, word2, word3))

        if predicate == 'pushable_obj':
            # Extract the object entity from the grounding
            # breakpoint()
            obj = operators["grounding_Python"][2]
            results.append(pushable_obj(state, obj))


        if predicate == 'overlapping':
            results.append(overlapping(state, operators["grounding_Python"][0], operators["grounding_Python"][1], operators["grounding_Python"][2], operators["grounding_Python"][3]))

        if predicate == 'not overlapping':
            overlap = overlapping(state, operators["grounding_Python"][0], operators["grounding_Python"][1], operators["grounding_Python"][2], operators["grounding_Python"][3])
            results.append(negate(overlap))

        # if predicate == 'not at':
        # # Extract words from grounding_Python
        #     results.append(negate(at(state, operators["grounding_Python"][0], operators["grounding_Python"][1])))

        # if predicate == 'at':
        # # Extract words from grounding_Python
        #     results.append(at(state, operators["grounding_Python"][0], operators["grounding_Python"][1]))
        #     breakpoint()

        # instance level
        if predicate == 'not at':
            results.append(negate(at(state, operators["grounding_Python"][0], operators["grounding_Python"][2], operators["grounding_Python"][1])))

        if predicate == 'at':
            # breakpoint()
            # obj, loc, index = operators["grounding_Python"][0], operators["grounding_Python"][2], operators["grounding_Python"][1]
            # # breakpoint()
            # # Check if the object was pushed into a sink (you can customize the "sink" logic)
            # if (obj == "rock_obj" and state[obj] == [[8, 4], []]) or  (obj == "rock_obj" and state[obj] == [[], [8, 4]]):
            #     # state[obj] 
            #     breakpoint()
            #     # If the object is logically at the sink, but removed from the state, return True
            #     results.append(True)
            results.append(at(state, operators["grounding_Python"][0], operators["grounding_Python"][2], operators["grounding_Python"][1]))
            # breakpoint()

    # if operators["operator"] == 'push_to':
    #     breakpoint()
    # if operators["operator"] == 'break_rule':
    #     breakpoint()
    # breakpoint()
    print("condition evalutions list:", results)
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
