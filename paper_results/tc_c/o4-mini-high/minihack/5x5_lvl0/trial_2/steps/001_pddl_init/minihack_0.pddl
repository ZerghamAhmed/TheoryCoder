(define (problem dungeon-problem)
  (:domain dungeon)
  (:objects
    human_rogue_called_agent staircase_down staircase_up - object
  )
  (:init
    ;; agent has not yet descended the down staircase
    (not (descended human_rogue_called_agent staircase_down))
  )
  (:goal
    ;; agent wins by descending
    (descended human_rogue_called_agent staircase_down)
  )
)