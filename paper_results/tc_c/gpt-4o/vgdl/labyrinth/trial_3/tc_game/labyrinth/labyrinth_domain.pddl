(define (domain grid-game)
  (:requirements :strips :typing)

  (:types 
    agent goal - object
  )

  (:predicates
    (reached ?a - agent ?g - goal)
  )

  (:action reachgoal
    :parameters (?a - agent ?g - goal)
    :precondition (not (reached ?a ?g))
    :effect (reached ?a ?g)
  )
)