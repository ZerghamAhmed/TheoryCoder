(define (problem pickup-problem)
  (:domain pickup-domain)

  (:objects
    red_agent - agent
    grey_key yellow_key purple_key red_key - key
    purple_door_1 blue_door_1 blue_door_2 red_door_1 red_door_2 red_door_3 yellow_door_1 grey_door_1 - door
  )

  (:init
    ;; initially no keys are carried and no doors are opened
  )

  (:goal
    (and
      ;; collect all keys
      (carrying red_agent grey_key)
      (carrying red_agent yellow_key)
      (carrying red_agent purple_key)
      (carrying red_agent red_key)
      ;; open all doors
      (opened purple_door_1)
      (opened blue_door_1)
      (opened blue_door_2)
      (opened red_door_1)
      (opened red_door_2)
      (opened red_door_3)
      (opened yellow_door_1)
      (opened grey_door_1)
    )
  )
)