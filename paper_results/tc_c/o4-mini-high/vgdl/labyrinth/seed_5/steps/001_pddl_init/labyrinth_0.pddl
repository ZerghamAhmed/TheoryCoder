(define (problem simplegrid-problem)
  (:domain simplegrid)

  (:objects
    avatar goal - object
  )

  (:init
    (not (reaches avatar goal))
  )

  (:goal
    (reaches avatar goal)
  )
)