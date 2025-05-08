(define (domain labyrinth)
    (:requirements :strips :negative-preconditions :equality :conditional-effects :typing)

    (:types 
      object 
    )

    (:predicates
        (overlapping ?obj1 - object ?obj2 - object)
    )

    (:action move_to
        :parameters (?obj - object ?to)
        :precondition (not (overlapping ?obj ?to))
        :effect (overlapping ?obj ?to)
    )

)
