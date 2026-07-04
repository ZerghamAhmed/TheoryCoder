(define (problem grid-game-problem)
  (:domain grid-game)

  (:objects
    avatar - agent
    hole - goal
  )

  (:init
    ;; Initial state: the goal (hole) has not been reached by the avatar
    (not (reached avatar hole))
  )

  (:goal
    ;; Goal: the avatar must reach the hole
    (reached avatar hole)
  )
)