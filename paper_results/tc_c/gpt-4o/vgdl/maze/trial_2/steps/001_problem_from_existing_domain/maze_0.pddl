(define (problem win-game-problem)
  (:domain win-game)

  (:objects
    avatar - avatar
    cheese - goal
  )

  (:init
    ;; Initially the avatar is not over the cheese
    (not (avatarover cheese))
  )

  (:goal
    ;; Goal is for the avatar to move onto the cheese
    (avatarover cheese)
  )
)