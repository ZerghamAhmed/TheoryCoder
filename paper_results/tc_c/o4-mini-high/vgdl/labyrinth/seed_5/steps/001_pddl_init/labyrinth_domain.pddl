(define (domain simplegrid)
  (:requirements :strips :typing)

  (:types
    object
  )

  (:predicates
    (reaches ?x - object ?y - object)
  )

  (:action reachgoal
    :parameters (?a - object ?b - object)
    :precondition (not (reaches ?a ?b))
    :effect (reaches ?a ?b)
  )
)