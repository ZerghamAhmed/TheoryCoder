(define (domain toy-domain)
  (:requirements :strips :typing)
  (:types
    object
  )
  (:predicates
    (killed ?x - object ?y - object)
  )

  (:action kill
    :parameters (?agent - object ?minotaur - object)
    :precondition (not (killed ?agent ?minotaur))
    :effect (killed ?agent ?minotaur)
  )
)