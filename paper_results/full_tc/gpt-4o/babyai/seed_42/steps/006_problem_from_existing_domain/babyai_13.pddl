(define (problem open-red-door)
  (:domain grid-game)

  (:objects
    red_agent - agent
    red_key - key
    red_door_1 - door
  )

  (:init
    ;; Red key is not carried initially
    (not (carrying red_agent red_key))
  )

  (:goal
    (door-unlocked red_door_1)
  )
)