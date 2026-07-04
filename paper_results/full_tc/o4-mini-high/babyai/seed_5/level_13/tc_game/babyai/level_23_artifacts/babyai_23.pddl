(define (problem pickup-problem-level2)
  (:domain pickup-domain)

  (:objects
    red_agent   - agent
    red_key     - key
    red_door_1  - door
  )

  (:init
    ;; agent does not yet have the key
    (not (has red_agent red_key))
  )

  (:goal
    (open red_door_1)
  )
)