(define (domain dungeon)
  (:requirements :strips :typing)
  (:types
    object
  )
  (:predicates
    (descended ?a - object ?s - object)
  )
  (:action descend
    :parameters (?agent - object ?stair - object)
    :precondition (not (descended ?agent ?stair))
    :effect (descended ?agent ?stair)
  )
)