(define (problem dungeon-problem)
  (:domain dungeon-domain)

  (:objects
    human_rogue_called_agent staircase_down - object
  )

  (:init
    ;; Initially the agent is not on the staircase
    (not (ontop human_rogue_called_agent staircase_down))
  )

  (:goal
    (ontop human_rogue_called_agent staircase_down)
  )
)