(define (domain doordomain)
  (:requirements :strips :typing)
  (:types
    door
  )
  (:predicates
    (opened ?d - door)
  )
  (:action opendoor
    :parameters (?d - door)
    :precondition (not (opened ?d))
    :effect (opened ?d)
  )
)