(define (problem grid-game-problem)
  (:domain grid-game)

  (:objects
    wall avatar hole box - object
  )

  (:init
    ;; Assuming the initial positions don't have specific ontop relationships
    (not (ontop box hole))
  )

  (:goal
    (ontop box hole)
  )
)
