(define (domain door-domain)
  (:requirements :strips :typing)
  (:types
    door
  )
  (:predicates
    (open ?d - door)
  )

  (:action open_door
    :parameters (?d - door)
    :precondition (not (open ?d))
    :effect (open ?d)
  )
)