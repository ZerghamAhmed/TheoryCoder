(define (domain baba)
    (:requirements :strips :negative-preconditions :equality :conditional-effects :typing)

    (:types 
        word object location
    )

    (:predicates
        (control_rule ?obj_name - object ?word2 - word ?word3 - word)
        (at ?obj - object ?loc - location)
        (overlapping ?obj1 - object ?obj2 - object)
        (rule_formed ?word1 - word ?word2 - word ?word3 - word)
        (rule_formable ?word1 - word ?word2 - word ?word3 - word)
        (rule_breakable ?word1 - word ?word2 - word ?word3 - word)
        (pushable_obj ?obj - object)
    )


    (:action move_to
        :parameters (?obj - object ?to)
        :precondition (and (control_rule ?obj is_word you_word) (not (overlapping ?obj ?to)) )
        :effect (overlapping ?obj ?to)
    )

    (:action push_to
        :parameters (?pusher - object ?obj - object ?to)
        :precondition (and (not (overlapping ?obj ?to)) (pushable_obj ?obj) (control_rule ?pusher is_word you_word) (not (overlapping ?pusher ?to)))
        :effect (and (overlapping ?obj ?to) (not (overlapping ?pusher ?to)))
    )

    (:action form_rule
        :parameters (?word1 - word ?word2 - word ?word3 - word)
        :precondition (and (not (rule_formed ?word1 ?word2 ?word3)) (rule_formable ?word1 ?word2 ?word3))
        :effect (rule_formed ?word1 ?word2 ?word3)
    )

    (:action break_rule
        :parameters (?word1 - word ?word2 - word ?word3 - word)
        :precondition (and (rule_formed ?word1 ?word2 ?word3) (rule_breakable ?word1 ?word2 ?word3))
        :effect (not (rule_formed ?word1 ?word2 ?word3))
    )

)
