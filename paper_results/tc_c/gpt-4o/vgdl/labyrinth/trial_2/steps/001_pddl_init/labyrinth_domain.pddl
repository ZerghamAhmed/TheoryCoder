(define (domain win-game)
  (:requirements :strips :typing)

  (:types
    avatar goal - object
  )

  (:predicates
    (avatarover ?x - goal)
  )

  (:action moveonto
    :parameters (?a - avatar ?g - goal)
    :precondition (not (avatarover ?g))
    :effect (avatarover ?g)
  )
)