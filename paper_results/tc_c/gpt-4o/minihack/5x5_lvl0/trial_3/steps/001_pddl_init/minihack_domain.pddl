(define (domain dungeon-domain)
  (:requirements :strips :typing)

  (:types
    object
  )

  (:predicates
    (descended ?x - object ?y - object)
  )

  (:action descend
    :parameters (?obj1 - object ?obj2 - object)
    :precondition (not (descended ?obj1 ?obj2))
    :effect (descended ?obj1 ?obj2)
  )
)