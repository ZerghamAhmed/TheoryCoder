(define (problem pickup-problem)
  (:domain pickup-domain)

  (:objects
    red_agent   - agent
    red_key     - key
    red_door_1  - door
  )

  (:init
    (not (pickedup   red_agent red_key))
    (not (unlocked  red_door_1))
  )

  (:goal
    (unlocked red_door_1)
  )
)