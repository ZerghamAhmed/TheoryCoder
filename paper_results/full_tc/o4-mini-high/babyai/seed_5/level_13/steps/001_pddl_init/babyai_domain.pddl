(define (domain pickup-domain)
  (:requirements :strips :typing)

  (:types
    agent
    key
  )

  (:predicates
    (has ?a - agent ?k - key)
  )

  (:action pickup
    :parameters (?a - agent ?k - key)
    :precondition (not (has ?a ?k))
    :effect (has ?a ?k)
  )
)