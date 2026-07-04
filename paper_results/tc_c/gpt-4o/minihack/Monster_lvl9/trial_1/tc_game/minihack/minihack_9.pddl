(define (problem dungeon-problem)
  (:domain dungeon-domain)

  (:objects
    kobold_zombie grid_bug staircase_down goblin human_rogue_called_agent - object
  )

  (:init
    ;; Initially the agent has not descended the staircase
    (not (descended human_rogue_called_agent staircase_down))
  )

  (:goal
    (descended human_rogue_called_agent staircase_down)
  )
)