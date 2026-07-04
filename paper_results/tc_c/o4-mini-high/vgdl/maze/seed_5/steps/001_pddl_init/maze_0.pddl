(define (problem reachproblem)
  (:domain reachdomain)

  (:objects
    avatar cheese - object
  )

  (:init
    (not (overlaps avatar cheese))
  )

  (:goal
    (overlaps avatar cheese)
  )
)