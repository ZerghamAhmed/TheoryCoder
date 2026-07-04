(define (problem grid-problem)
  (:domain grid-domain)

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