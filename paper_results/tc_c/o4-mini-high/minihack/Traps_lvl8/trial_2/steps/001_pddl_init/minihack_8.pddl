(define (problem dungeonproblem)
  (:domain dungeondomain)

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