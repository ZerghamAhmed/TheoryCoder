(define (domain pushdomain)
  (:requirements :strips :typing)
  (:types object)
  (:predicates
    (overlaps ?b - object ?h - object)
  )
  (:action push
    :parameters (?b - object ?h - object)
    :precondition (not (overlaps ?b ?h))
    :effect (overlaps ?b ?h)
  )
)