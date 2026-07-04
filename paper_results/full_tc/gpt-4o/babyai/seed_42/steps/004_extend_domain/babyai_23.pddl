(define (problem grid-game-problem)
  (:domain grid-game)

  (:objects
    red_agent - agent
    red_key - key
    red_door_1 - door
  )

  (:init
    (not (carrying red_agent red_key))
    (not (door-unlocked red_door_1))
  )

  (:goal
    (and
      (door-unlocked red_door_1)
    )
  )
)