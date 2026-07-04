(define (problem gridproblem)
  (:domain gridgame)

  (:objects
    avatar cheese - object
  )

  (:init
    (not (touching avatar cheese))
  )

  (:goal
    (touching avatar cheese)
  )
)