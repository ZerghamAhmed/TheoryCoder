(define (domain gridgame-domain)
  (:requirements :strips :typing)
  (:types object)
  (:predicates
    (overlaps ?x - object ?y - object)
  )
  (:action reaches
    :parameters (?x - object ?y - object)
    :precondition (not (overlaps ?x ?y))
    :effect (overlaps ?x ?y)
  )
)