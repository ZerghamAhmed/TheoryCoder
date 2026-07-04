(define (problem open-yellow-door-problem)
  (:domain grid-game)

  (:objects
    red_agent - agent
    yellow_key - key
    yellow_door_1 - door
  )

  (:init
    ;; Agent is not carrying the key initially
    (not (carrying red_agent yellow_key))
  )

  (:goal
    ;; The yellow door must be unlocked
    (unlocked yellow_door_1)
  )
)