(define (problem pickup-problem)
  (:domain pickup-domain)
  (:objects
    red_door_1 - door
  )
  (:init
    (not (opened red_door_1))
  )
  (:goal
    (and
      (opened red_door_1)
    )
  )
)