(define (problem dungeon-problem)
  (:domain dungeon-domain)

  (:objects
    human_rogue_called_agent staircase_down - object
  )

  (:init
    ;; Initially the agent has not descended the staircase
    (not (descended human_rogue_called_agent staircase_down))
  )

  (:goal
    ;; The goal is to have the agent descend the staircase
    (descended human_rogue_called_agent staircase_down)
  )
)