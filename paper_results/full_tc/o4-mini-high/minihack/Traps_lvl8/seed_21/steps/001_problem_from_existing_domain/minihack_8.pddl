(define (problem rogue-problem)
  (:domain rogue-domain)

  (:objects
    human_rogue_called_agent staircase_down - object
  )

  (:init
    ; initially the agent has not descended the staircase
  )

  (:goal
    (descended human_rogue_called_agent staircase_down)
  )
)