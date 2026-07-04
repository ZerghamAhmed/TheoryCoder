(define (domain rogue-domain)
  (:requirements :strips :typing)

  (:types
    object
  )

  (:predicates
    (descended ?x - object ?y - object)
  )

  (:action descend
    :parameters (?agent - object ?stair - object)
    :precondition (not (descended ?agent ?stair))
    :effect (descended ?agent ?stair)
  )
)