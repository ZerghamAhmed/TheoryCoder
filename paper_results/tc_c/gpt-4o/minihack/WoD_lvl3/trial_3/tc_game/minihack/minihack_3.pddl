(define (problem game-problem)
  (:domain game-domain)

  (:objects
    agent minotaur wand staircase_down - object
  )

  (:init
    ;; Initially, the minotaur is not killed by the agent
    (not (killed agent minotaur))
  )

  (:goal
    (killed agent minotaur)
  )
)