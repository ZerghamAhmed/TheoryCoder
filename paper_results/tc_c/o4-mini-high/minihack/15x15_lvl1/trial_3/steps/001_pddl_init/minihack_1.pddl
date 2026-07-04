(define (problem dungeon-problem)
  (:domain dungeon)

  (:objects
    human_rogue_called_agent staircase_down staircase_up inventory won lost - object
  )

  (:init
    ;; initially the agent is not overlapping the down staircase
    (not (overlap human_rogue_called_agent staircase_down))
  )

  (:goal
    ;; overlapping the down staircase means we've descended (won)
    (overlap human_rogue_called_agent staircase_down)
  )
)