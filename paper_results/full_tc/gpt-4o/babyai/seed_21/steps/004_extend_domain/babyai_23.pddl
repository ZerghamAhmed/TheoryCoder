(define (problem grid-game-problem)
  (:domain grid-game)

  (:objects
    red_agent - agent
    red_key - key
    red_door_1 - door
  )

  (:init
    (not (carrying red_agent red_key)) ;; Agent is not initially carrying the key
    (not (unlocked red_door_1)) ;; Door is locked
  )

  (:goal
    (and
      (unlocked red_door_1) ;; Goal is to unlock the door
    )
  )
)