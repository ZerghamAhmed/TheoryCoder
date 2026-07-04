(define (problem rogue-problem)
  (:domain rogue-domain)
  (:objects
    human_rogue_called_agent staircase_down - object
  )
  (:init
    ;; agent is not yet on the stairs (closed‐world: overlap is false by omission)
  )
  (:goal
    (and
      (overlap human_rogue_called_agent staircase_down)
    )
  )
)