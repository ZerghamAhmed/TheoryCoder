(define (domain descend_domain)
  (:requirements :strips :typing)

  (:types
    object
  )

  (:predicates
    (descended ?a - object ?b - object)
  )

  (:action descend
    :parameters (?a - object ?b - object)
    :precondition (not (descended ?a ?b))
    :effect (descended ?a ?b)
  )
)