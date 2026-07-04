(define (domain pickup-domain)
  (:requirements :strips :typing)
  (:types
    agent
    key
    door
  )
  (:predicates
    (carrying ?a - agent ?k - key)
    (opened ?d - door)
  )

  (:action pickup
    :parameters (?a - agent ?k - key)
    :precondition (not (carrying ?a ?k))
    :effect (carrying ?a ?k)
  )

  (:action open_door
    :parameters (?d - door)
    :precondition (not (opened ?d))
    :effect (opened ?d)
  )
)