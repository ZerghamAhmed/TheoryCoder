(define (problem dungeon-problem)
  (:domain dungeon-domain)
  (:objects
    kobold human_rogue_called_agent lichen grid_bug staircase_down - object
  )
  (:init
    (not (descended human_rogue_called_agent staircase_down))
  )
  (:goal
    (descended human_rogue_called_agent staircase_down)
  )
)