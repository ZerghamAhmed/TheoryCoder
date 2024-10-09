(define (domain baba)
    (:requirements :strips :negative-preconditions :equality :conditional-effects :typing)

    (:types 
        word word_instance object_instance location orientation
    )

    (:predicates
        (control_rule ?obj_name - object_instance ?word2 - word ?word3 - word)
        (push_rule ?obj_name - object_instance ?word2 - word ?word3 - word)
        (at ?obj - object_instance ?loc - location)
        (overlapping ?obj1 - object_instance ?obj2 - object_instance)
        (rule_formed ?word1 - word_instance ?word2 - word_instance ?word3 - word_instance)
    )


    (:action move_to
        :parameters (?obj - object_instance ?to)
        :precondition (and (control_rule ?obj is you) (not (overlapping ?obj ?to)) )
        :effect (overlapping ?obj ?to)
    )

    (:action move_loc
        :parameters (?obj - object_instance ?to)
        :precondition (and (control_rule ?obj is you) (not (at ?obj ?to)) )
        :effect (at ?obj ?to)
    )

    (:action push_to
        :parameters (?obj ?to)
        :precondition (and (not (at ?obj ?to)) )
        :effect (at ?obj ?to)
    )

    (:action form_rule
        :parameters (?word1 - word_instance ?word2 - word_instance ?word3 - word_instance)
        :precondition (not (rule_formed ?word1 ?word2 ?word3))
        :effect (rule_formed ?word1 ?word2 ?word3)
    )

    (:action break_rule
        :parameters (?word1 - word_instance ?word2 - word_instance ?word3 - word_instance)
        :precondition (rule_formed ?word1 ?word2 ?word3)
        :effect (not (rule_formed ?word1 ?word2 ?word3))
    )

)
