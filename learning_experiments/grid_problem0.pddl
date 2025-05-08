(define (problem grid-game-problem)
  (:domain grid-game-domain)

  (:objects
    avatar goal - object
  )

  (:init
    ;; Initially avatar is not on top of the goal
    (not (ontop avatar goal))
  )

  (:goal
    (ontop avatar goal)
  )
)
