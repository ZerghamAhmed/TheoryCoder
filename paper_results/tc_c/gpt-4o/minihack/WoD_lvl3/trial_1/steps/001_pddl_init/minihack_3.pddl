(define (problem minotaur-problem)
  (:domain minotaur-domain)

  (:objects
    agent minotaur wand - object
  )

  (:init
    (not (killed agent minotaur))
  )

  (:goal
    (killed agent minotaur)
  )
)