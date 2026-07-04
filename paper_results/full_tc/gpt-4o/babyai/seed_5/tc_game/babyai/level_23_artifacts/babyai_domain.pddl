(define (domain grid-game-domain)
  (:requirements :strips :typing)

  (:types
    agent
    key
    door
  )

  (:predicates
    (carrying ?a - agent ?k - key)
    (door_unlocked ?d - door)
  )

  (:action pickkey
    :parameters (?a - agent ?k - key)
    :precondition (not (carrying ?a ?k))
    :effect (carrying ?a ?k)
  )

  (:action unlockdoor
    :parameters (?a - agent ?k - key ?d - door)
    :precondition (carrying ?a ?k)
    :effect (door_unlocked ?d)
  )
)