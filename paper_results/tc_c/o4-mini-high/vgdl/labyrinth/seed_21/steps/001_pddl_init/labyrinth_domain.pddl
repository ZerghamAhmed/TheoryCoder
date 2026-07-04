(define (domain reach-domain)
  (:requirements :strips :typing)
  (:types object)
  (:predicates
    (overlaps ?x - object ?y - object)
  )
  (:action reachgoal
    :parameters (?a - object ?g - object)
    :precondition (not (overlaps ?a ?g))
    :effect (overlaps ?a ?g)
  )
)