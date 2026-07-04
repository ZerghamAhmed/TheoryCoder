(define (domain sokoban)
  (:requirements :strips :typing)
  (:types object)
  (:predicates
    (overlaps ?x - object ?y - object)
  )
  (:action push
    :parameters (?b - object ?h - object)
    :precondition (not (overlaps ?b ?h))
    :effect (overlaps ?b ?h)
  )
)