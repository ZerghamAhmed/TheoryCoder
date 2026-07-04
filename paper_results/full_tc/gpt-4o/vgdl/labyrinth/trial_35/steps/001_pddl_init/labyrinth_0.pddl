(define (problem grid-problem)
  (:domain grid-domain)

  (:objects
    avatar goal - object
  )

  (:init
    ;; Initially, the avatar has not reached the goal
    (not (reached avatar goal))
  )

  (:goal
    ;; The goal is for the avatar to reach the goal
    (reached avatar goal)
  )
)