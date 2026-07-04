(define (domain dungeondomain)
  (:requirements :strips :typing)

  (:types
    object
  )

  (:predicates
    (descended ?agent - object ?staircase - object)
  )

  (:action descend
    :parameters (?agent - object ?staircase - object)
    :precondition (not (descended ?agent ?staircase))
    :effect (descended ?agent ?staircase)
  )
)