(define (domain grid-domain)
  (:requirements :strips :typing)

  (:types
    object
  )

  (:predicates
    (reached ?x - object ?y - object)
  )

  (:action reach
    :parameters (?agent - object ?target - object)
    :precondition (not (reached ?agent ?target))
    :effect (reached ?agent ?target)
  )
)