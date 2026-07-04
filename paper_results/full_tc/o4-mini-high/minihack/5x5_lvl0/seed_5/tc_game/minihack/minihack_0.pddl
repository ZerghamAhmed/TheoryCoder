(define (problem rogue-problem)
  (:domain rogue-domain)
  (:objects
    human_rogue_called_agent staircase_down - object
  )
  (:init
    ;; agent is not yet on the downstairs staircase
    (not (overlap human_rogue_called_agent staircase_down))
  )
  (:goal
    ;; reaching the downstairs staircase wins the game
    (overlap human_rogue_called_agent staircase_down)
  )
)