(define (domain simple-domain)
  (:requirements :strips :typing)
  (:types object)
  (:predicates
    (killed ?agent - object ?minotaur - object)
  )
  (:action kill
    :parameters (?agent - object ?minotaur - object)
    :precondition (not (killed ?agent ?minotaur))
    :effect (killed ?agent ?minotaur)
  )
)