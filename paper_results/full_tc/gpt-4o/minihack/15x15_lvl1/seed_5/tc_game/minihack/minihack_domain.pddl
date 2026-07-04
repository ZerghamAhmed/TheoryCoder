(define (domain dungeon-domain)
  (:requirements :strips :typing)

  (:types
    object
  )

  (:predicates
    (descended ?x - object)
  )

  (:action descend
    :parameters (?obj - object)
    :precondition (not (descended ?obj))
    :effect (descended ?obj)
  )
)