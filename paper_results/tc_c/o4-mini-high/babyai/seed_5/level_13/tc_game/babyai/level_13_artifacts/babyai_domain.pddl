(define (domain game-domain)
  (:requirements :strips :typing)
  (:types
    door
  )
  (:predicates
    (opened ?d - door)
  )
  (:action open-door
    :parameters (?d - door)
    :precondition (not (opened ?d))
    :effect (opened ?d)
  )
)