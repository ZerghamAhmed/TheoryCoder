(define (domain open_domain)
  (:requirements :strips :typing :negative-preconditions)
  (:types
    door
  )
  (:predicates
    (opened ?d - door)
  )
  (:action open_door
    :parameters (?d - door)
    :precondition (not (opened ?d))
    :effect (opened ?d)
  )
)