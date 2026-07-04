(define (problem grid-game-problem)
  (:domain grid-game-domain)

  (:objects
    red_agent - agent
    yellow_key - key
  )

  (:init
    ;; Initially the agent is not carrying the key
    (not (carrying red_agent yellow_key))
  )

  (:goal
    ;; The agent must be carrying the key
    (carrying red_agent yellow_key)
  )
)