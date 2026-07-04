(define (problem game-problem)
  (:domain game-domain)
  (:objects
    red_door_1 - door
  )
  (:init
    (not (opened red_door_1))
  )
  (:goal
    (opened red_door_1)
  )
)