(define (domain rogue-domain)
  (:requirements :strips :typing)
  (:types
    object
  )
  (:predicates
    (overlap ?x - object ?y - object)
  )
  (:action descend
    :parameters (?agent - object ?stairs - object)
    :precondition (not (overlap ?agent ?stairs))
    :effect (overlap ?agent ?stairs)
  )
)