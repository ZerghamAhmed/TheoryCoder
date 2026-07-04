(define (domain reach-goal-domain)
  (:requirements :strips :typing)

  (:types
    object
  )

  (:predicates
    (reached ?x - object ?y - object)
  )

  (:action reach
    :parameters (?agent - object ?goal - object)
    :precondition (not (reached ?agent ?goal))
    :effect (reached ?agent ?goal)
  )
)