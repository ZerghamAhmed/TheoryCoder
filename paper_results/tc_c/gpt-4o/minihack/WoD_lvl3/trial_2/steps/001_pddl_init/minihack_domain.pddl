(define (domain minotaur-game)
  (:requirements :strips :typing)

  (:types
    object
  )

  (:predicates
    (killed ?attacker - object ?target - object)
  )

  (:action slay
    :parameters (?attacker - object ?target - object)
    :precondition (and (not (killed ?attacker ?target)))
    :effect (killed ?attacker ?target)
  )
)