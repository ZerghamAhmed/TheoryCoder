(define (problem pickup-problem)
  (:domain pickup-domain)

  (:objects
    red_agent - agent
    grey_key red_key green_key yellow_key - key
    red_door_1 green_door_1 green_door_2
    yellow_door_1 yellow_door_2 yellow_door_3 yellow_door_4 yellow_door_5
    grey_door_1 - door
  )

  (:init
    ;; Doors that are initially unlocked (locked = false in the raw state)
    (unlocked green_door_1)
    (unlocked green_door_2)
    (unlocked yellow_door_1)
    (unlocked yellow_door_2)
    (unlocked yellow_door_3)
    (unlocked yellow_door_4)
    (unlocked yellow_door_5)
    (unlocked grey_door_1)
  )

  (:goal
    (and
      ;; Pick up the red key
      (pickedup red_agent red_key)
      ;; Unlock the only locked door (red_door_1) to win
      (unlocked red_door_1)
    )
  )
)