(define (domain pickup-domain)
  (:requirements :strips :typing)
  (:types
    agent
    key
  )
  (:predicates
    (carrying ?a - agent ?k - key)
  )
  (:action pickup
    :parameters (?a - agent ?k - key)
    :precondition (not (carrying ?a ?k))
    :effect (carrying ?a ?k)
  )
)