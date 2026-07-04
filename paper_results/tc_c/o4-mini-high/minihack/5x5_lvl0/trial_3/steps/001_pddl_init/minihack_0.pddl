(define (problem rogue-problem)
  (:domain rogue-domain)
  (:objects
    human_rogue_called_agent staircase_down staircase_up inventory - object
  )
  (:init
    ;; agent has not yet descended
    (not (overlap human_rogue_called_agent staircase_down))
  )
  (:goal
    ;; agent overlaps the down staircase => descended and wins
    (overlap human_rogue_called_agent staircase_down)
  )
)