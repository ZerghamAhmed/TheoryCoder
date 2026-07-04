(define (problem grid-problem)
  (:domain grid-game)

  (:objects
    avatar - agent
    goal - goal
  )

  (:init
    ;; Initially, the avatar has not reached the goal
    (not (reached avatar goal))
  )

  (:goal
    ;; Goal is for the avatar to reach the goal object
    (reached avatar goal)
  )
)