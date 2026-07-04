(define (problem descend-problem)
  (:domain descend-domain)

  (:objects
    staircase_down human_rogue_called_agent - object
  )

  (:init
    ;; Initially, the agent has not descended the staircase
    (not (descended human_rogue_called_agent staircase_down))
  )

  (:goal
    (descended human_rogue_called_agent staircase_down)
  )
)