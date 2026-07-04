(define (problem dungeon-problem)
  (:domain dungeon-domain)

  (:objects
    human_rogue_called_agent staircase_down - object
  )

  (:init
    ;; Initially the agent has not descended
    (not (descended human_rogue_called_agent staircase_down))
  )

  (:goal
    ;; The goal is for the agent to have descended
    (descended human_rogue_called_agent staircase_down)
  )
)