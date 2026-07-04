(define (problem dungeon-problem)
  (:domain dungeon)

  (:objects
    human_rogue_called_agent staircase_down staircase_up inventory won lost - object
  )

  (:init
    (not (descended human_rogue_called_agent staircase_down))
  )

  (:goal
    (descended human_rogue_called_agent staircase_down)
  )
)