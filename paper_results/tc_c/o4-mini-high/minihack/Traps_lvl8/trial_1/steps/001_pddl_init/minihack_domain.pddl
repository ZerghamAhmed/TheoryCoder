(define (domain dungeon-domain)
  (:requirements :strips :typing :negative-preconditions)

  (:types
    object
  )

  (:predicates
    (won ?x - object)
  )

  (:action descend
    :parameters (?x - object)
    :precondition (not (won ?x))
    :effect (won ?x)
  )
)