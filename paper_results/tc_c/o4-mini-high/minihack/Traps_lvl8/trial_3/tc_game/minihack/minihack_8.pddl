(define (problem descend_problem)
  (:domain descend_domain)

  (:objects
    human_rogue_called_agent staircase_down - object
  )

  (:init
    (not (descended human_rogue_called_agent staircase_down))
  )

  (:goal
    (descended human_rogue_called_agent staircase_down)
  )
)