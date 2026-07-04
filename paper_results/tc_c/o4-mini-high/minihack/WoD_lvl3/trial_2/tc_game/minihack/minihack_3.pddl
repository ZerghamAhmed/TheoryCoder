(define (problem simple-problem)
  (:domain simple-domain)
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