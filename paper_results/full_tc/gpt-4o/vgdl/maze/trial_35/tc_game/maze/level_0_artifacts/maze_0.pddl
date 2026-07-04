(define (problem grid-problem)
  (:domain grid-domain)

  (:objects
    avatar - object
    cheese - object
  )

  (:init
    (not (reached avatar cheese))
  )

  (:goal
    (reached avatar cheese)
  )
)