(define (problem grid-game-problem)
  (:domain grid-game)

  (:objects
    avatar - agent
    cheese - goal
  )

  (:init
    ;; Initial condition: avatar has not reached the cheese yet
    (not (reached avatar cheese))
  )

  (:goal
    ;; Goal: avatar reaches the cheese
    (reached avatar cheese)
  )
)