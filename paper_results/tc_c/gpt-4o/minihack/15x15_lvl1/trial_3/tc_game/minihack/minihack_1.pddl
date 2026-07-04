(define (problem descend-problem)
  (:domain descend-dungeon)

  (:objects
    human_rogue_called_agent staircase_down - object
  )

  (:init
    ;; Initially nothing has descended
    (not (descended human_rogue_called_agent staircase_down))
  )

  (:goal
    (descended human_rogue_called_agent staircase_down)
  )
)