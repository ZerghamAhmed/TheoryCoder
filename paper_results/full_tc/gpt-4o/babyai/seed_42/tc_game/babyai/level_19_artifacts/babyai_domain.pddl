(define (domain grid-game)
  (:requirements :strips :typing)

  (:types
    agent key object
  )

  (:predicates
    (carrying ?a - agent ?k - key)
  )

  (:action pickkey
    :parameters (?a - agent ?k - key)
    :precondition (not (carrying ?a ?k))
    :effect (carrying ?a ?k)
  )
)