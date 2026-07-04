(define (domain pickup-domain)
  (:requirements :strips :typing)

  (:types
    agent
    key
    door
  )

  (:predicates
    (has ?a - agent ?k - key)
    (open ?d - door)
  )

  (:action pickup
    :parameters (?a - agent ?k - key)
    :precondition (not (has ?a ?k))
    :effect (has ?a ?k)
  )

  (:action open_door
    :parameters (?a - agent ?k - key ?d - door)
    :precondition (has ?a ?k)
    :effect (open ?d)
  )
)