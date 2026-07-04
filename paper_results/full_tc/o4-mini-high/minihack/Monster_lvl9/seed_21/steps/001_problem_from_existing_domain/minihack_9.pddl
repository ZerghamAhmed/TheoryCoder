(define (problem rogue-descend-problem)
  (:domain rogue-domain)

  (:objects
    human_rogue_called_agent staircase_down - object
  )

  (:init
    ;; no initial descended facts; by default (descended ...) is false
  )

  (:goal
    (and
      ;; subgoal: agent must be able to descend (i.e. not yet descended)
      ;; achieved by the descend action
      (descended human_rogue_called_agent staircase_down)
    )
  )
)