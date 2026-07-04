(define (domain dungeon-domain)
  (:requirements :strips :typing)

  (:types
    object
  )

  (:predicates
    (descended ?x - object ?y - object)
  )

  (:action descend
    :parameters (?hero - object ?stair - object)
    :precondition (not (descended ?hero ?stair))
    :effect (descended ?hero ?stair)
  )
)