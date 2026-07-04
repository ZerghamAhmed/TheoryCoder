(define (domain grid-game)
  (:requirements :strips :typing)

  (:types
    agent key door object
  )

  (:predicates
    (carrying ?a - agent ?k - key)
    (door-unlocked ?d - door)
  )

  (:action pickkey
    :parameters (?a - agent ?k - key)
    :precondition (not (carrying ?a ?k))
    :effect (carrying ?a ?k)
  )

  (:action unlockdoor
    :parameters (?a - agent ?k - key ?d - door)
    :precondition (and (carrying ?a ?k))
    :effect (door-unlocked ?d)
  )
)