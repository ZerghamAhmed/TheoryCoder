(define (domain reachdomain)
  (:requirements :strips :typing)
  (:types
    object
  )
  (:predicates
    (overlaps ?x - object ?y - object)
  )
  (:action reach
    :parameters (?a - object ?c - object)
    :precondition (not (overlaps ?a ?c))
    :effect (overlaps ?a ?c)
  )
)