(define (domain dungeon)
  (:requirements :strips :typing :negative-preconditions)
  (:types object)
  (:predicates
    (overlap ?x - object ?y - object)
  )
  (:action descend
    :parameters (?human_rogue_called_agent - object
                 ?staircase_down            - object)
    :precondition (not (overlap ?human_rogue_called_agent ?staircase_down))
    :effect (overlap ?human_rogue_called_agent ?staircase_down)
  )
)