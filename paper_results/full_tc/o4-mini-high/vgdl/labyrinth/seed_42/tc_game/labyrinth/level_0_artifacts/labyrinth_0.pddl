(define (problem grid-problem)
  (:domain grid-domain)
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