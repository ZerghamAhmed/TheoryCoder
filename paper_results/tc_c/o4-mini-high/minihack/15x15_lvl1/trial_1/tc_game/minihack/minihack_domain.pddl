(define (domain dungeon)
  (:requirements :strips :typing)

  (:types
    object
  )

  (:predicates
    (overlap ?x - object ?y - object)
  )

  (:action descend
    :parameters (?h - object ?s - object)
    :precondition (not (overlap ?h ?s))
    :effect (overlap ?h ?s)
  )
)