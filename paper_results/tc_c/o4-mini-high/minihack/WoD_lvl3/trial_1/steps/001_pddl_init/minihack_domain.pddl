(define (domain grid-domain)
  (:requirements :strips :typing)

  (:types
    object
  )

  (:predicates
    (killed ?a - object ?b - object)
  )

  (:action kill
    :parameters (?a - object ?b - object)
    :precondition (not (killed ?a ?b))
    :effect (killed ?a ?b)
  )
)