(define (domain dungeon-domain)
  (:requirements :strips :typing)

  (:types
    object
  )

  (:predicates
    (descended ?x - object ?y - object)
  )

  (:action descend
    :parameters (?agent - object ?staircase - object)
    :precondition (not (descended ?agent ?staircase))
    :effect (descended ?agent ?staircase)
  )
)