(define (problem open-purple-door-problem)
  (:domain grid-game-domain)

  (:objects
    red_agent - agent
    purple_key - key
    purple_door_1 - door
  )

  (:init
    ;; The agent is not carrying the purple key initially
    (not (carrying red_agent purple_key))
  )

  (:goal
    (door_unlocked purple_door_1)
  )
)