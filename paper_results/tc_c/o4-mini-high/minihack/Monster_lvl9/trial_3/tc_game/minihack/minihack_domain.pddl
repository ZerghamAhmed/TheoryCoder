(define (domain dungeon)
  (:requirements :strips :typing)

  (:types
    object
  )

  (:predicates
    (overlap ?x - object ?y - object)
    (won)
  )

  (:action descend
    :parameters (?human_rogue_called_agent - object ?staircase_down - object)
    :precondition (overlap ?human_rogue_called_agent ?staircase_down)
    :effect (won)
  )
)