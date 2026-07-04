(define (problem toy-problem)
  (:domain toy-domain)
  (:objects
    agent minotaur - object
  )
  (:init
    (not (killed agent minotaur))
  )
  (:goal
    (killed agent minotaur)
  )
)