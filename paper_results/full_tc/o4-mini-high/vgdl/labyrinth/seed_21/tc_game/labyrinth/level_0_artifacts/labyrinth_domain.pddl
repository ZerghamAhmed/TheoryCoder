(define (domain reachgame)
  (:requirements :strips :typing)
  (:types object)
  (:predicates
    (colocated ?x - object ?y - object)
  )
  (:action reach
    :parameters (?a - object ?b - object)
    :precondition (not (colocated ?a ?b))
    :effect (colocated ?a ?b)
  )
)