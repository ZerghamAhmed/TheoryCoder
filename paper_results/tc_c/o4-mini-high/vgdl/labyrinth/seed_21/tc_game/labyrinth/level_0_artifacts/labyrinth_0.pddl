(define (problem reach-problem)
  (:domain reach-domain)

  (:objects
    avatar goal - object
  )

  (:init
    (not (overlaps avatar goal))
  )

  (:goal
    (overlaps avatar goal)
  )
)