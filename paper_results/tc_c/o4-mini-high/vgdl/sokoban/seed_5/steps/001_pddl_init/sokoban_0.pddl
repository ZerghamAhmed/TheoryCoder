(define (problem pushproblem)
  (:domain pushdomain)
  (:objects
    box hole - object
  )
  (:init
    (not (overlaps box hole))
  )
  (:goal
    (overlaps box hole)
  )
)