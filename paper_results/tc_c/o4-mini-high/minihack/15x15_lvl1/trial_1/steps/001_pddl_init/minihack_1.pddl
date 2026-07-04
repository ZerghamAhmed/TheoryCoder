(define (problem dungeon-problem)
  (:domain dungeon)

  (:objects
    human_rogue_called_agent staircase_down staircase_up inventory won lost - object
  )

  (:init
    (not (overlap human_rogue_called_agent staircase_down))
  )

  (:goal
    (overlap human_rogue_called_agent staircase_down)
  )
)