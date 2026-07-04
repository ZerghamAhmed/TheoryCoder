(define (problem grid-problem)
  (:domain grid-domain)

  (:objects
    wall avatar hole box floor - object
  )

  (:init
    ;; no reaches facts initially
  )

  (:goal
    ;; every box must overlap a hole
    (reaches box hole)
  )
)