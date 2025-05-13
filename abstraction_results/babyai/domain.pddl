(define (domain gridgame-domain)
  (:requirements :strips :typing)

  (:types
    object
  )

  (:predicates
    (holding ?a - object ?b - object)
  )

  (:action pickup
    :parameters (?a - object ?b - object)
    :precondition (not (holding ?a ?b))
    :effect (holding ?a ?b)
  )
)
