(define (problem open-purple-problem)
  (:domain game-domain)
  (:objects
    purple_door_1 - door
  )
  (:init
    (not (opened purple_door_1))
  )
  (:goal
    (opened purple_door_1)
  )
)