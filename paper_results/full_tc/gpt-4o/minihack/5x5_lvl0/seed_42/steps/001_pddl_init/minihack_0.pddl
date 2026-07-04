(define (problem dungeon-problem)
  (:domain dungeon-domain)

  (:objects
    human_rogue_called_agent staircase_down staircase_up - object
  )

  (:init
    ;; Initially, the hero has not descended.
    (not (descended human_rogue_called_agent staircase_down))
  )

  (:goal
    (descended human_rogue_called_agent staircase_down)
  )
)