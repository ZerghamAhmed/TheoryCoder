(define (domain grid-game-domain)
  (:requirements :strips :typing)
  
  (:types
    object
  )

  (:predicates
    (ontop ?x - object ?y - object)
  )

  (:action MoveToGoal
    :parameters (?obj1 - object ?obj2 - object)
    :precondition (not (ontop ?obj1 ?obj2))
    :effect (ontop ?obj1 ?obj2)
  )
)
