(define (domain dungeon-domain)
  (:requirements :strips :typing)

  (:types
    object
  )

  (:predicates
    (descended ?x - object ?y - object)
  )

  (:action descend
    :parameters (?agent - object ?stairs - object)
    :precondition (not (descended ?agent ?stairs))
    :effect (descended ?agent ?stairs)
  )
)