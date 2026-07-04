(define (problem reachgame-problem)
  (:domain reachgame)
  (:objects
    avatar goal - object
  )
  (:init
    (not (colocated avatar goal))
  )
  (:goal
    (colocated avatar goal)
  )
)