(define (domain grid-domain)
  (:requirements :strips :typing)
  (:types
    object
  )
  (:predicates
    (reaches ?x - object ?y - object)
  )
  (:action move
    :parameters (?agent - object ?target - object)
    :precondition (not (reaches ?agent ?target))
    :effect (reaches ?agent ?target)
  )
)