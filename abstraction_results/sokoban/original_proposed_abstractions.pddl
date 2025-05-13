(define (domain game-domain)
  (:requirements :strips :typing)

  (:types
    object
  )

  (:predicates
    (ontop ?x - object ?y - object)
  )

  (:action overlap
    :parameters (?x - object ?y - object)
    :precondition (not (ontop ?x ?y))
    :effect (ontop ?x ?y)
  )
)
