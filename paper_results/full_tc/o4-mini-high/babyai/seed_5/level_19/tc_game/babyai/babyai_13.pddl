(define (problem pickup-problem)
  (:domain pickup-domain)

  (:objects
    red_agent - agent
    purple_key grey_key yellow_key blue_key - key
    blue_door_1 blue_door_2 grey_door_1 grey_door_2
    yellow_door_1 yellow_door_2 green_door_1 green_door_2
    purple_door_1 - door
  )

  (:init
    ;; no keys in hand, no doors open (closed world assumption)
  )

  (:goal
    (and
      ;; pick up the purple key...
      (has red_agent purple_key)
      ;; ...and open the exit (purple) door
      (open purple_door_1)
    )
  )
)