(define (problem open_problem)
  (:domain open_domain)
  (:objects
    red_door_1 - door
  )
  (:init
    ;; no doors are opened initially
  )
  (:goal
    (opened red_door_1)
  )
)