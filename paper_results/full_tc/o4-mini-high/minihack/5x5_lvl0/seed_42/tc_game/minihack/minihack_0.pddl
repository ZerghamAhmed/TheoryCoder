(define (problem dungeon-problem)
  (:domain dungeon)

  (:objects
    human_rogue_called_agent staircase_up - object
  )

  (:init
    ;; the agent is overlapping the up‐stairs
    (overlap human_rogue_called_agent staircase_up)
  )

  (:goal
    (won human_rogue_called_agent)
  )
)