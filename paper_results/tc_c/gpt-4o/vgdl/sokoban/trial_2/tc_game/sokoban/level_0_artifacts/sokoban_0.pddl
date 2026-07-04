(define (problem win-game-problem)
  (:domain win-game)

  (:objects
    avatar - avatar
    hole - goal
  )

  (:init
    ;; Initially, the avatar is NOT over the goal
    (not (avatarover hole))
  )

  (:goal
    ;; Goal: Avatar must end up over the hole
    (avatarover hole)
  )
)