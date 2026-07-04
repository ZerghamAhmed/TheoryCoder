(define (problem game-problem)
  (:domain game-domain)
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