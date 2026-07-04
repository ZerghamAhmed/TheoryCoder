(define (problem reach-cheese-problem)
  (:domain reachgame)

  (:objects
    avatar cheese - object
  )

  (:init
    ;; no objects are initially colocated
  )

  (:goal
    (colocated avatar cheese)
  )
)