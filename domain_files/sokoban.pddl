(define (domain sokoban)
  (:requirements :strips :typing)

  (:types
    box avatar
  )

  (:predicates
    (unstored_box ?box - box)   ; The box is found (exists in the game)
    (at_goal ?agent - avatar) ; Avatar is at the goal (optional for further goals)
    (boxes_stuck)              ; Checks if any box in the state is stuck

  )

  (:action push_to_hole
    :parameters (?box - box)
    :precondition (unstored_box ?box)
    :effect (and (not (unstored_box ?box)) (not (boxes_stuck)))
  )
)
