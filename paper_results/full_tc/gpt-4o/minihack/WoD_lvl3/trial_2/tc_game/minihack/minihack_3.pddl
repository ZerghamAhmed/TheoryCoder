(define (problem minotaur-problem)
  (:domain minotaur-game)

  (:objects
    agent minotaur wand - object
  )

  (:init
    ;; Initially the minotaur is not killed
    (not (killed agent minotaur))
  )

  (:goal
    ;; The agent needs to kill the minotaur
    (killed agent minotaur)
  )
)