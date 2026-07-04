(define (domain open-domain)
  (:requirements :strips :typing)
  (:types
    door
  )
  (:predicates
    (open ?d - door)
  )
  (:action open-door
    :parameters (?d - door)
    :effect (open ?d)
  )
)