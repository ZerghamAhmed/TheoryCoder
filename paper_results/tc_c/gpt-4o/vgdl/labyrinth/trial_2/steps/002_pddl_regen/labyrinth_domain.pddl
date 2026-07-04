(define (domain win-game)
  (:requirements :strips :typing)

  (:types
    avatar goal - object
  )

  (:predicates
    ;; Predicate to represent that the avatar is over the goal
    (avatarover ?g - goal)
  )

  (:action moveonto
    :parameters (?a - avatar ?g - goal)
    :precondition (not (avatarover ?g))
    :effect (and (avatarover ?g))
  )
)