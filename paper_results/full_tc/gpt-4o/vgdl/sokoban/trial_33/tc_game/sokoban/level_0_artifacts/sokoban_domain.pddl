(define (domain grid-domain)
  (:requirements :strips :typing)

  (:types
    object
  )

  (:predicates
    ;; Specifies that one object is on top of another
    (ontop ?x - object ?y - object)
  )

  (:action move
    :parameters (?obj1 - object ?obj2 - object)
    :precondition (not (ontop ?obj1 ?obj2))
    :effect (ontop ?obj1 ?obj2)
  )
)