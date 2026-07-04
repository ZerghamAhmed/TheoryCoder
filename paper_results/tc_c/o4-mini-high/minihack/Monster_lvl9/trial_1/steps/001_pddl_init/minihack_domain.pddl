(define (domain dungeon-domain)
  (:requirements :strips :typing)

  (:types
    object
  )

  (:predicates
    (descended ?agent - object ?stair - object)
  )

  (:action descend
    :parameters (?agent - object ?stair - object)
    :precondition (not (descended ?agent ?stair))
    :effect (descended ?agent ?stair)
  )
)