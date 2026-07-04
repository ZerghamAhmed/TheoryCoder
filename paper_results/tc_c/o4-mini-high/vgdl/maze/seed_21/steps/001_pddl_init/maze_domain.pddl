(define (domain gridgame)
  (:requirements :strips :typing)
  (:types object)

  (:predicates
    (touching ?x - object ?y - object)
  )

  (:action touch
    :parameters (?x - object ?y - object)
    :precondition (not (touching ?x ?y))
    :effect (touching ?x ?y)
  )
)