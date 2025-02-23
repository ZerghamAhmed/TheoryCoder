(define (domain push_boulders_1)
  (:requirements :strips :typing)

  (:types
    location avatar object 
    poison2 
    poison1 
    box1 
    box2 
    dynamite 
    goal
  )

  (:predicates
    (at ?place) ; Agent is at a specific location
    (connection ?from  ?to) ; Path exists between two locations
  )

  (:action move_between_bottleneck
    :parameters (?from - object ?to - object)
    :precondition (and (at ?from) (connection ?from ?to))
    :effect (and (at ?to))
  )

)
