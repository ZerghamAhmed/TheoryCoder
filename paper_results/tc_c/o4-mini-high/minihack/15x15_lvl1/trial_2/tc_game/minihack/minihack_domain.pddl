(define (domain descend-domain)
  (:requirements :strips :typing)
  (:types
    object
  )
  (:predicates
    (won)
  )
  (:action descend
    :parameters (?x - object)
    :precondition (not (won))
    :effect (won)
  )
)