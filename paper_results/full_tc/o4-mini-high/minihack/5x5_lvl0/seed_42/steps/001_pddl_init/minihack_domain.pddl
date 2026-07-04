(define (domain dungeon)
  (:requirements :strips :typing)

  (:types
    object
  )

  (:predicates
    (overlap ?x - object ?y - object)
    (won ?x - object)
  )

  (:action descend
    :parameters (?a - object ?s - object)
    :precondition (overlap ?a ?s)
    :effect (won ?a)
  )
)