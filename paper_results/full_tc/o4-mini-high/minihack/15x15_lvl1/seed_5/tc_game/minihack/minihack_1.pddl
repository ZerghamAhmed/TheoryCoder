(define (problem rogue-problem)
  (:domain rogue-domain)

  (:objects
    human_rogue_called_agent staircase_down staircase_up - object
  )

  (:init
    ;; Initially the agent is not overlapping any staircase
  )

  (:goal
    (and
      ;; The agent must visit both staircases
      (overlap human_rogue_called_agent staircase_up)
      (overlap human_rogue_called_agent staircase_down)
    )
  )
)