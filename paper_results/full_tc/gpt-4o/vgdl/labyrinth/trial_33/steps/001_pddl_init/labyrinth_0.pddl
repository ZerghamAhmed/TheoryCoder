(define (problem grid-problem)
  (:domain grid-domain)

  (:objects
    avatar goal - object
  )

  (:init
    ;; Initially, the avatar is not on the goal
    (not (ontop avatar goal))
  )

  (:goal
    ;; The avatar wins when it is on top of the goal
    (ontop avatar goal)
  )
)