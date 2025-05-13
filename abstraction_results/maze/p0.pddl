(define (problem grid-game-problem)
  (:domain grid-game)

  (:objects
    wall floor cheese rat - object
  )

  (:init
    ;; Initially, the rat is not on top of the cheese
    (not (ontop rat cheese))
  )

  (:goal
    ;; The goal is to have the rat on top of the cheese
    (ontop rat cheese)
  )
)
