(define (problem open-yellow-door-problem)
  (:domain door-domain)

  (:objects
    yellow_door_1 - door
  )

  (:init
    (not (open yellow_door_1))
  )

  (:goal
    (open yellow_door_1)
  )
)