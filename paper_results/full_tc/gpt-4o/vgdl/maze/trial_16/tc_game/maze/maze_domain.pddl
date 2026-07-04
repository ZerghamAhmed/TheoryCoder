(define (domain grid-game)
  (:requirements :strips :typing)

  (:types
    object
  )

  (:predicates
    ;; Represents the avatar being at the goal
    (avatargoal ?x - object ?y - object)
  )

  (:action reachgoal
    :parameters (?avatar - object ?goal - object)
    :precondition (not (avatargoal ?avatar ?goal))
    :effect (avatargoal ?avatar ?goal)
  )
)