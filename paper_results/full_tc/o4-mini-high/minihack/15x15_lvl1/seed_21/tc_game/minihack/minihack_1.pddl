(define (problem rogue-descend-problem)
  (:domain rogue-domain)

  (:objects
    human_rogue_called_agent staircase_down staircase_up - object
  )

  (:init
    ;; No initial 'descended' facts; by default nothing is descended.
  )

  (:goal
    (and
      ;; The agent must descend both staircases as subgoals
      (descended human_rogue_called_agent staircase_up)
      (descended human_rogue_called_agent staircase_down)
    )
  )
)