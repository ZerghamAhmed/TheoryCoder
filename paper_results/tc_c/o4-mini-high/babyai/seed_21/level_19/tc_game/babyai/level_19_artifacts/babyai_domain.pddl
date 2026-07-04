(define (domain pick-domain)
  (:requirements :strips :typing)

  (:types
    agent
    key
  )

  (:predicates
    (pickedup ?ag - agent ?k - key)
  )

  (:action pick
    :parameters (?ag - agent ?k - key)
    :precondition (not (pickedup ?ag ?k))
    :effect (pickedup ?ag ?k)
  )
)