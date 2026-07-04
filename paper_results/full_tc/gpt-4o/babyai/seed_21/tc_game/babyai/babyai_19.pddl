(define (problem grid-problem)
  (:domain grid-game)

  (:objects
    red_agent - agent
    yellow_key - key
  )

  (:init
    ;; Initially the agent is not carrying the key
    (not (carrying red_agent yellow_key))
  )

  (:goal
    ;; The agent must carry the key
    (carrying red_agent yellow_key)
  )
)