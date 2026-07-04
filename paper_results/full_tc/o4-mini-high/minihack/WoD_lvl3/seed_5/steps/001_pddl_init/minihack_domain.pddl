(define (domain game-domain)
  (:requirements :strips :typing)

  (:types
    object
  )

  (:predicates
    (killed ?x - object ?y - object)
  )

  (:action kill
    :parameters (?attacker - object ?target - object)
    :precondition (not (killed ?attacker ?target))
    :effect (killed ?attacker ?target)
  )
)