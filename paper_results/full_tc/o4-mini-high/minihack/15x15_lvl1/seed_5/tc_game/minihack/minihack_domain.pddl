(define (domain rogue-domain)
  (:requirements :strips :typing)
  (:types
    object
  )
  (:predicates
    (overlap ?x - object ?y - object)
  )
  (:action gotostair
    :parameters (?agent - object ?stair - object)
    :precondition (not (overlap ?agent ?stair))
    :effect (overlap ?agent ?stair)
  )
)