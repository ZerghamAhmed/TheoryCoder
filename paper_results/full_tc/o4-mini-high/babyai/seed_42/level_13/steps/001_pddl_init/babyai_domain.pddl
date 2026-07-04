(define (domain pickup-domain)
  (:requirements :strips :typing)

  (:types
    agent
    key
  )

  (:predicates
    (pickedup ?a - agent ?k - key)
  )

  (:action pickup
    :parameters (?a - agent ?k - key)
    :precondition (not (pickedup ?a ?k))
    :effect (pickedup ?a ?k)
  )
)