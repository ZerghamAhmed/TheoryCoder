(define (problem dungeon-problem)
  (:domain dungeon-domain)

  (:objects
    human_rogue_called_agent staircase_down staircase_up - object
  )

  (:init
    ;; The agent hasn't descended yet
    (not (descended human_rogue_called_agent staircase_down))
  )

  (:goal
    ;; The agent should descend the staircase
    (descended human_rogue_called_agent staircase_down)
  )
)