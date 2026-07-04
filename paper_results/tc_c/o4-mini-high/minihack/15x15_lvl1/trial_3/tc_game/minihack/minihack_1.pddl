(define (problem dungeon-problem)
  (:domain dungeon)
  (:objects
    human_rogue_called_agent staircase_down staircase_up inventory won lost - object
  )
  (:init
    ;; no initial overlap; default is false
  )
  (:goal
    (overlap human_rogue_called_agent staircase_down)
  )
)