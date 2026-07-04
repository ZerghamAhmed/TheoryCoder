(define (problem win-game-problem)
  (:domain win-game)

  (:objects
    avatar - avatar
    goal - goal
  )

  (:init
    ;; Initially, the avatar is NOT over the goal
    (not (avatarover goal))
  )

  (:goal
    ;; The goal is to make the avatar overlap the goal object
    (avatarover goal)
  )
)