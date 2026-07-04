(define (domain pickup-domain)
  (:requirements :strips :typing)

  (:types
    agent
    key
    door
  )

  (:predicates
    (pickedup ?a - agent ?k - key)
    (unlocked ?d - door)
  )

  (:action pickup
    :parameters (?a - agent ?k - key)
    :precondition (not (pickedup ?a ?k))
    :effect (pickedup ?a ?k)
  )

  (:action open_door
    :parameters (?a - agent ?k - key ?d - door)
    :precondition (pickedup ?a ?k)
    :effect (unlocked ?d)
  )
)