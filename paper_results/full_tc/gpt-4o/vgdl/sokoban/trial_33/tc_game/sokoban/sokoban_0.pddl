(define (problem grid-problem)
  (:domain grid-domain)

  (:objects
    wall avatar hole box floor - object
  )

  (:init
    ;; Initial state: box is not on top of the hole
    (not (ontop box hole))
  )

  (:goal
    ;; Goal: box must be on top of the hole
    (ontop box hole)
  )
)