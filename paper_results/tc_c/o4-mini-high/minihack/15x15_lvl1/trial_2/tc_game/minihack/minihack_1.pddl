(define (problem descend-problem)
  (:domain descend-domain)
  (:objects
    human_rogue_called_agent staircase_down staircase_up - object
  )
  (:init
    (not (won))
  )
  (:goal
    (won)
  )
)