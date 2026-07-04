(define (domain game-domain)
  (:requirements :strips :typing)
  (:types
    object
  )
  (:predicates
    (reaches ?x - object ?y - object)
  )
  (:action reach
    :parameters (?x - object ?y - object)
    :precondition (not (reaches ?x ?y))
    :effect (reaches ?x ?y)
  )
)