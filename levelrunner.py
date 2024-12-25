import re
import os
from collections import deque
from predicates import *
from preprocessing import operator_extractor, checker
from copy import deepcopy
from games import BabaIsYou
from babareport import BabaReportUpdater
from planner import enumerative_search
import planner

def actor(domain_file, subplan, state, max_iterations=None, debug_callback=None, level=None):
    operator = operator_extractor(domain_file, subplan)

    print("OPERATOR AND PRECONDS LIST", operator)

    preconditions = operator['preconditions']
    effects = operator['effects']

    # breakpoint()

    # Evaluate preconditions checker(state, predicates, operator)
    precondition_results = checker(state, preconditions, operator)
    if not precondition_results:
        unsatisfied_preconds = [pre for pre in operator["preconditions"] if not checker(state, [pre], operator)]
        return f"Preconditions not satisfied to be able to execute this {subplan}. Unsatisfied preconditions: {unsatisfied_preconds}", state

    else:
        print(f"Preconditions satisfied to be able to execute {subplan}")
        actions, new_state = enumerative_search(state, operator, preconditions, effects, debug_callback=debug_callback, level=level)
        if actions:
            return actions, new_state

def process_level(level, subplans, domain_file, updater, max_iterations=None, checkpoint_dir='checkpoints'):
    engine = BabaIsYou(level_set='demo_LEVELS', level_id=level)
    state = engine.get_obs()
    state = {key: [list(item) for item in value] if isinstance(value, list) else value for key, value in state.items()}

    controllables = {
                    entity for entity in state
                    if rule_formed(state, f'{entity[:-4]}_word', 'is_word', 'you_word')
                }

        
        # Add controllables to the state dictionary
    state['controllables'] = list(controllables)

    overlappables = {
                    entity for entity in state
                    if rule_formed(state, f'{entity[:-4]}_word', 'is_word', 'win_word') 
                    or entity.endswith('goop_obj')
                }
        
    pushables = {
                entity for entity in state
                if entity.endswith('_word')
            }

    
    # Add controllables to the state dictionary
    state['controllables'] = list(controllables)

    if 'empty' in state:
        del state['empty']

    if 'won' in state:
        del state['won']

    state['overlappables'] = list(overlappables)

    #  Add controllables to the state dictionary
    state['pushables'] = list(pushables)

    word_entities = [entity for entity in state.keys() if entity.endswith('_word')]
    rules_on_map = []
    for subj in word_entities:
        for pred in word_entities:
            for obj in word_entities:
                if rule_formed(state, subj, pred, obj):
                    # print(f"Rule formed: {subj} {pred} {obj}")
                    rules_on_map.append(subj + ' ' + pred + ' ' + obj)


    state['rules_formed'] = rules_on_map

    breakpoint()
    
    actions_full = []
    first_letters = ''

    for subplan in subplans:
        actions, state = actor(domain_file, subplan, state, max_iterations, level=level)
        print(f"Actions to achieve the goal ({subplan}):", actions)
        print("New state:", state)

        for action in actions:
            first_letters += action[0]
        
        actions_full.append(actions)

    print("FINAL ACTIONS", actions_full)
    print(first_letters)

    updater.update_solution(level_id=level+1, first_letters=first_letters)

    # Create checkpoint directory if it doesn't exist
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Write checkpoint file
    checkpoint_file = os.path.join(checkpoint_dir, f'level_{level}_updated.txt')
    with open(checkpoint_file, 'w') as f:
        f.write(f'Level {level} updated with actions: {first_letters}\n')
        f.write(f'Actions: {actions_full}\n')

if __name__ == "__main__":
    domain_file = 'domain.pddl'
    json_report_path = 'KekeCompetition-main/Keke_JS/reports/TBRL_BABA_REPORT.json'
    updater = BabaReportUpdater(json_report_path)

    # optionally clear all levels
    # updater.clear_solutions()

    # breakpoint()

    # Define subplans for each level
    level_subplans = {
        0: ["move_to baba_obj_1 flag_obj_1"],
        1: ["form_rule flag_word_1 is_word_2 win_word_1"],
        # 2: ["push_to win_word_1 [4,5]", "push_to is_word_1 [4,6]"],
        2: ["form_rule baba_word_1 is_word_1 win_word_1"],

        3: ["move_to flag_obj_1 baba_obj_1"],
        4: ["form_rule rock_word_1 is_word_3 flag_word_2"],
        5: ["break_rule rock_word_1 is_word_2 rock_word_2", "move_to baba_obj_1 flag_obj_1"],
        6: ["move_to baba_obj_1 flag_obj_1"],
        7: ["move_to baba_obj_1 flag_obj_1"],
        8: ["break_rule wall_word_1 is_word_1 stop_word_1", "move_to baba_obj_1 flag_obj_1"],
        9: ["break_rule skull_word_1 is_word_1 kill_word_1", "move_to baba_obj_1 flag_obj_1"],
        10: ["break_rule goop_word_1 is_word_3 sink_word_1", "move_to baba_obj_1 flag_obj_1"],
        # 11: ["move_loc baba_obj_1 [8,3]"], worked when pair this with baba obj to flag obj but took like 2 mins
        11: ["move_loc baba_obj_1 [8,2]", "move_to baba_obj_1 flag_obj_1"],

        12: ["break_rule baba_word_1 is_word_2 melt_word_1", "move_to baba_obj_1 flag_obj_1"],
        13: ["form_rule keke_word_1 is_word_1 you_word_1", "move_to keke_obj_1 flag_obj_1"]
    }

    # Directory for checkpoints
    checkpoint_dir = 'checkpoints'

    # List of levels to run ID - 1 
    # levels_to_run = [2]
    # levels_to_run = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

    # # levels_to_run = [11]
    # # levels_to_run = [2]
    # # levels_to_run = [7]
    # levels_to_run = [2]
    # levels_to_run = [11]
    # # levels_to_run = [7]
    # levels_to_run = [1]
    # levels_to_run = [7]
    levels_to_run = [10]
    # breakpoint()

    for level in levels_to_run:
        process_level(level, level_subplans[level], domain_file, updater, max_iterations=100)
