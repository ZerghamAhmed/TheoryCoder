(define (problem toy-problem)
  (:domain toy-domain)

  (:objects
    agent minotaur wand - object
  )

  (:init
    ; Initially, the agent has not killed the minotaur
    (not (killed agent minotaur))
  )

  (:goal
    ; The goal is for the agent to kill the minotaur
    (killed agent minotaur)
  )
)