(define (domain grid-game)
  (:requirements :strips :typing)

  (:types
    agent
    key
    door
  )

  (:predicates
    (carrying ?a - agent ?k - key)
    (unlocked ?d - door) ;; New predicate for door state
  )

  (:action pickup
    :parameters (?a - agent ?k - key)
    :precondition (not (carrying ?a ?k))
    :effect (carrying ?a ?k)
  )

  (:action unlock
    :parameters (?a - agent ?k - key ?d - door)
    :precondition (carrying ?a ?k) ;; Requires the agent to be carrying the key
    :effect (unlocked ?d) ;; Marks the door as unlocked
  )
)