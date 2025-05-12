(define (problem grid-problem)
  (:domain grid-game)

  (:objects
    wall avatar floor goal - object
  )

  (:init
    ;; Initially, the avatar is not on top of the goal
    (not (ontop avatar goal))
  )

  (:goal
    (ontop avatar goal)
  )
)
