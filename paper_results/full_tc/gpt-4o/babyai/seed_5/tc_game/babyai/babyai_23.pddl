(define (problem grid-game-level2)
  (:domain grid-game-domain)

  (:objects
    red_agent - agent
    red_key - key
    red_door_1 - door
  )

  (:init
    (not (carrying red_agent red_key))
    (not (door_unlocked red_door_1))
  )

  (:goal
    (and
      (door_unlocked red_door_1)
    )
  )
)