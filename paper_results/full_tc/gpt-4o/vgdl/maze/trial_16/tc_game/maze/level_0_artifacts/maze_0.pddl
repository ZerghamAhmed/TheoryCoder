(define (problem grid-game-problem)
  (:domain grid-game)

  (:objects
    avatar - object
    cheese - object
  )

  (:init
    ;; Initially, the avatar is not at the cheese
    (not (avatargoal avatar cheese))
  )

  (:goal
    ;; Goal is for the avatar to reach the cheese
    (avatargoal avatar cheese)
  )
)